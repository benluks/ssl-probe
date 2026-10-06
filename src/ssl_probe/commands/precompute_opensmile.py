# scripts/precompute_opensmile.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import opensmile
from quick_convert.data.loading import load_dataset
from tqdm import tqdm

from ..opensmile import OPENSMILE_LLD_FRAME_HZ, init_opensmile, validate_jitter_min_periods
from ..paths import dataset_utt_id_overrides, utterance_relative_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits to process. Omit to use Quick Convert dataset defaults or scan a generic root.",
    )
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
    return args


def main():
    args = parse_args()

    dataset_kwargs = dataset_utt_id_overrides(args.dataset, args.utt_id_template)

    dataset = load_dataset(
        name=args.dataset,
        root=args.root,
        splits=args.splits,
        load=["audio"],
        **dataset_kwargs,
    )

    # Quick Convert owns split selection and validation. Generic root scans
    # have no split label and store metadata directly in the output folder.
    splits = dataset.splits if dataset.splits is not None else [None]

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
        split_dir = out_dir / split if split is not None else out_dir
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
        split_dir = out_dir / split if split is not None else out_dir
        (split_dir / "_metadata.json").write_text(json.dumps(metadata, indent=2))

    for sample in tqdm(
        dataset, desc="+".join(split for split in splits if split is not None) or "audio"
    ):
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
