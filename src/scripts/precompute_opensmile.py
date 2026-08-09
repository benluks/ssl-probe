# scripts/precompute_opensmile.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import opensmile
from quick_convert.data import load_dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--out-root",
        default="features/opensmile",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = load_dataset(
        "librispeech",
        root=args.root,
        splits=[args.split],
        load=["audio"],
    )

    feature_set = opensmile.FeatureSet.eGeMAPSv02
    feature_level = opensmile.FeatureLevel.LowLevelDescriptors

    smile = opensmile.Smile(
        feature_set=feature_set,
        feature_level=feature_level,
    )

    out_dir = Path(args.out_root) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "feature_set": feature_set.name,
        "feature_level": feature_level.name,
        "split": args.split,
    }

    (out_dir / "_metadata.json").write_text(json.dumps(metadata, indent=2))

    for sample in tqdm(dataset, desc=args.split):
        out_path = out_dir / f"{sample.utt_id}.parquet"

        if out_path.exists():
            continue

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
