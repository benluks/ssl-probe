# scripts/precompute_opensmile.py

from __future__ import annotations

import argparse
import json
import traceback
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import opensmile
from quick_convert.data.loading import load_dataset
from tqdm import tqdm

from ..opensmile import OPENSMILE_LLD_FRAME_HZ, init_opensmile, validate_jitter_min_periods
from ..paths import dataset_utt_id_overrides, utterance_relative_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--root")
    source.add_argument(
        "--dataset-config", type=Path, help="Trusted Hydra YAML with a dataset node."
    )
    parser.add_argument("--limit", type=int, help="Process at most this many samples.")
    parser.add_argument(
        "--on-error",
        choices=["raise", "skip"],
        default="raise",
        help="Stop on extraction errors (default), or record them and process remaining samples.",
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--splits", nargs="+")
    parser.add_argument(
        "--out-root",
        default="features/opensmile",
    )
    parser.add_argument("--out-folder", default=None)
    parser.add_argument(
        "--utt-id-template",
        default=None,
        help="Optional override. Named datasets use their Quick Convert template.",
    )

    parser.add_argument(
        "--jitter-min-periods",
        type=int,
        default=2,
        help="Minimum pitch periods for jitter/shimmer (>=2; default: 2).",
    )
    args = parser.parse_args(argv)
    try:
        validate_jitter_min_periods(args.jitter_min_periods)
    except ValueError as error:
        parser.error(str(error))
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.dataset_config is None and not args.splits:
        parser.error("--splits is required with --root")
    if args.dataset_config is not None:
        if args.dataset or args.splits or args.utt_id_template:
            parser.error("--dataset-config cannot be combined with dataset discovery options")
        if not args.out_folder:
            parser.error("--out-folder is required with --dataset-config")
    return args


def main():
    args = parse_args()

    if args.dataset_config is not None:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        config = OmegaConf.load(args.dataset_config)
        # Instantiate only the selected node: unused fold placeholders may be missing.
        dataset = instantiate(config.dataset, load=["audio"])
    else:
        dataset_kwargs = dataset_utt_id_overrides(args.dataset, args.utt_id_template)
        dataset = load_dataset(
            name=args.dataset,
            root=args.root,
            splits=args.splits,
            load=["audio"],
            **dataset_kwargs,
        )

    # Inspect metadata without decoding audio. Unsplit datasets store metadata at the root.
    rows = list(islice(dataset.rows, args.limit))
    if not rows:
        raise ValueError("The selected dataset contains no samples.")
    splits = list(dict.fromkeys(row.split for row in rows))
    seen = set()
    for row in rows:
        relative_path = utterance_relative_path(row.utt_id, row.split)
        if relative_path in seen:
            raise ValueError(f"Duplicate output ID: {relative_path}")
        seen.add(relative_path)
    dataset.rows = rows

    feature_set = opensmile.FeatureSet.eGeMAPSv02
    feature_level = opensmile.FeatureLevel.LowLevelDescriptors

    smile = init_opensmile(
        feature_set=feature_set,
        feature_level=feature_level,
        jitter_min_periods=args.jitter_min_periods,
    )

    out_folder = args.out_folder or args.dataset
    if out_folder is None:
        raise ValueError("you must set a dataset name in --dataset or pass --out-folder")

    out_dir = Path(args.out_root) / out_folder
    for split in splits:
        split_dir = out_dir if split is None else out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "feature_set": feature_set.name,
            "feature_level": feature_level.name,
            "frame_hz": OPENSMILE_LLD_FRAME_HZ,
            "jitter_min_periods": args.jitter_min_periods,
            "split": split,
        }

        metadata_path = split_dir / "_metadata.json"
        if metadata_path.exists():
            previous = json.loads(metadata_path.read_text())
            previous.setdefault("jitter_min_periods", 2)
            if previous != metadata:
                raise ValueError(
                    f"Extraction settings conflict at {metadata_path}; use a new folder."
                )
        elif any(split_dir.rglob("*.parquet")):
            raise ValueError(
                f"Existing features have no metadata at {split_dir}; use a new folder."
            )

    # Validate every split before updating any metadata.
    for split in splits:
        metadata = {
            "feature_set": feature_set.name,
            "feature_level": feature_level.name,
            "frame_hz": OPENSMILE_LLD_FRAME_HZ,
            "jitter_min_periods": args.jitter_min_periods,
            "split": split,
        }
        split_dir = out_dir if split is None else out_dir / split
        (split_dir / "_metadata.json").write_text(json.dumps(metadata, indent=2))

    run_id = uuid4().hex
    error_report = out_dir / "_errors.jsonl"
    summary = {
        "run_id": run_id,
        "selected": len(rows),
        "cached": 0,
        "written": 0,
        "failed": 0,
        "status": "interrupted",
    }
    description = "+".join(split if split is not None else "all" for split in splits)
    try:
        for index, row in enumerate(tqdm(rows, desc=description)):
            relative_path = utterance_relative_path(row.utt_id, row.split)
            out_path = out_dir / relative_path.parent / f"{relative_path.name}.parquet"
            # Check metadata before dataset access: __getitem__ decodes the audio.
            if out_path.exists():
                summary["cached"] += 1
                continue

            stage = "load"
            try:
                sample = dataset[index]
                stage = "opensmile"
                waveform = sample.waveform
                if waveform.ndim == 2 and waveform.shape[0] == 1:
                    waveform = waveform.squeeze(0)
                features = smile.process_signal(
                    waveform.detach().cpu().numpy(),
                    int(sample.sample_rate),
                ).reset_index()
            except Exception as error:
                summary["failed"] += 1
                failure = {
                    "run_id": run_id,
                    "time": datetime.now(UTC).isoformat(),
                    "index": index,
                    "utt_id": row.utt_id,
                    "path": str(row.path),
                    "split": row.split,
                    "start": row.start,
                    "end": row.end,
                    "output": str(out_path),
                    "stage": stage,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                # Append and close immediately so failures survive later interruptions.
                with error_report.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(failure) + "\n")
                tqdm.write(f"Failed {row.utt_id} ({stage}): {error}")
                if args.on_error == "raise":
                    raise RuntimeError(
                        f"Extraction failed for {row.utt_id!r}; details in {error_report}"
                    ) from error
                continue

            # Storage errors must still stop the run. Never leave a partial cache entry.
            out_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with NamedTemporaryFile(dir=out_path.parent, suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                features.to_parquet(temporary, index=False)
                temporary.replace(out_path)
                summary["written"] += 1
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        summary["status"] = "completed_with_errors" if summary["failed"] else "complete"
    finally:
        (out_dir / "_extraction_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Cached: {summary['cached']}; written: {summary['written']}; failed: {summary['failed']}."
    )
    if summary["failed"]:
        print(f"Extraction has missing samples; inspect {error_report} before training.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
