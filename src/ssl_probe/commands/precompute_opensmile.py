# scripts/precompute_opensmile.py

from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path

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

    for sample in tqdm(dataset, desc="+".join(str(split) for split in splits)):
        relative_path = utterance_relative_path(sample.utt_id, sample.split)
        out_path = out_dir / relative_path.parent / f"{relative_path.name}.parquet"

        if out_path.exists():
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)

        waveform = sample.waveform

        if waveform.ndim == 2 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)

        features = smile.process_signal(
            waveform.detach().cpu().numpy(),
            int(sample.sample_rate),
        )

        # openSMILE returns start/end as index levels.
        # Reset them so they are explicit columns in the parquet file.
        features = features.reset_index()

        features.to_parquet(
            out_path,
            index=False,
        )


if __name__ == "__main__":
    main()
