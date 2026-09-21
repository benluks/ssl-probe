# scripts/precompute_opensmile.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import opensmile
from quick_convert.data.loading import load_dataset
from tqdm import tqdm

from ..opensmile import init_opensmile


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument(
        "--out-root",
        default="features/opensmile",
    )
    parser.add_argument("--out-folder", default=None)
    parser.add_argument("--utt-id-template", default="{path.stem}")

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = load_dataset(
        name=args.dataset,
        root=args.root,
        splits=args.splits,
        load=["audio"],
        utt_id_template=args.utt_id_template,
    )

    feature_set = opensmile.FeatureSet.eGeMAPSv02
    feature_level = opensmile.FeatureLevel.LowLevelDescriptors

    smile = init_opensmile(
        feature_set=feature_set,
        feature_level=feature_level,
    )

    out_folder = args.out_folder or args.dataset
    if out_folder is None:
        raise ValueError("you must set a dataset name in --dataset or pass --out-folder")

    out_dir = Path(args.out_root) / out_folder
    for split in args.splits:
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "feature_set": feature_set.name,
            "feature_level": feature_level.name,
            "split": split,
        }

        (split_dir / "_metadata.json").write_text(json.dumps(metadata, indent=2))

    for sample in tqdm(dataset, desc="+".join(args.splits)):
        out_path = out_dir / sample.split / f"{sample.utt_id}.parquet"

        if out_path.exists():
            continue

        # out_path.parent.mkdir(parents=True, exist_ok=True)

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
