# scripts/precompute_opensmile.py

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import opensmile
from quick_convert.data.loading import load_dataset
from tqdm import tqdm

from ..opensmile import OPENSMILE_LLD_FRAME_HZ, init_opensmile
from ..paths import dataset_utt_id_overrides, utterance_relative_path

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits to process. Defaults to all split directories found under --root.",
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

    return parser.parse_args()


def find_splits(root: str | Path) -> list[str]:
    """Return the names of split directories available under ``root``."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"dataset root is not a directory: {root}")

    splits = sorted(path.name for path in root.iterdir() if path.is_dir())
    if not splits:
        raise ValueError(f"no split directories found under dataset root: {root}")
    return splits


def resolve_splits(root: str | Path, requested: list[str] | None) -> list[str]:
    available = find_splits(root)
    if requested is None:
        logger.info("Discovered %d splits under %s: %s", len(available), root, ", ".join(available))
        return available

    available_set = set(available)
    missing = [split for split in requested if split not in available_set]
    if missing:
        logger.warning(
            "Skipping requested splits not present under %s: %s",
            root,
            ", ".join(missing),
        )

    splits = [split for split in requested if split in available_set]
    if not splits:
        raise ValueError(f"none of the requested splits are present under dataset root: {root}")
    return splits


def main():
    args = parse_args()
    splits = resolve_splits(args.root, args.splits)

    dataset_kwargs = dataset_utt_id_overrides(args.dataset, args.utt_id_template)

    dataset = load_dataset(
        name=args.dataset,
        root=args.root,
        splits=splits,
        load=["audio"],
        **dataset_kwargs,
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
    for split in splits:
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "feature_set": feature_set.name,
            "feature_level": feature_level.name,
            "frame_hz": OPENSMILE_LLD_FRAME_HZ,
            "split": split,
        }

        (split_dir / "_metadata.json").write_text(json.dumps(metadata, indent=2))

    for sample in tqdm(dataset, desc="+".join(splits)):
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
