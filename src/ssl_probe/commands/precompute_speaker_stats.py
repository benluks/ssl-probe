from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from quick_convert.data import load_dataset
from quick_convert.data.resources import TemplateResourceProvider
from tqdm import tqdm

from ..targets.store import SmileParquetStore
from ..targets.target import LOG_F0, semitone_to_log_hz, valid_pitch_class


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--splits", nargs="+", required=True)

    parser.add_argument(
        "--features-root",
        default="features/opensmile",
    )
    parser.add_argument(
        "--features-folder",
        required=True,
    )

    parser.add_argument(
        "--out-root",
        default="features",
    )
    parser.add_argument(
        "--out-folder",
        default="speaker_stats",
    )

    parser.add_argument("--utt-id-template", default="{path.stem}")
    parser.add_argument("--spk-id-template", default="{path.parent.parent.stem}")

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = load_dataset(
        name=args.dataset,
        root=args.root,
        splits=args.splits,
        load=[],
        utt_id_template=args.utt_id_template,
        additional_resource_providers=[
            TemplateResourceProvider(
                name="spk_id",
                template=args.spk_id_template,
                kind="text",
            )
        ],
    )

    features_root = Path(args.features_root) / args.features_folder

    stores = {
        split: SmileParquetStore(
            smile_root=features_root / split,
            feature_column=LOG_F0.source,
        )
        for split in args.splits
    }

    sums = defaultdict(float)
    counts = defaultdict(int)

    for sample in tqdm(dataset, desc="+".join(args.splits)):
        semitone = stores[sample.split][sample.utt_id]

        valid = valid_pitch_class(semitone)
        semitone = semitone[valid]

        if semitone.numel() == 0:
            continue

        log_f0 = semitone_to_log_hz(semitone)

        sums[sample.resources["spk_id"].value] += log_f0.sum().item()
        counts[sample.resources["spk_id"].value] += log_f0.numel()

    speakers = {
        str(spk_id): {
            "mean_log_f0": sums[spk_id] / counts[spk_id],
            "voiced_frames": counts[spk_id],
        }
        for spk_id in sorted(counts)
    }

    output = {
        "dataset": args.dataset,
        "splits": args.splits,
        "source": {
            "features_folder": args.features_folder,
            "feature_column": LOG_F0.source,
        },
        "speakers": speakers,
    }

    out_dir = Path(args.out_root) / args.features_folder / args.out_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{args.dataset}.json"
    out_path.write_text(json.dumps(output, indent=2))

    print(f"Wrote speaker statistics for {len(speakers)} speakers to {out_path}")


if __name__ == "__main__":
    main()
