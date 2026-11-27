"""Audit saved acoustic targets without extracting audio or encoder features."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..paths import utterance_relative_path
from ..targets import TARGETS, RegressionTask


class Moments:
    """Merge population moments for raw and transformed values."""

    def __init__(self):
        self.count = 0
        self.mean = np.zeros(2)
        self.m2 = np.zeros(2)
        self.minimum = np.full(2, np.inf)
        self.maximum = np.full(2, -np.inf)

    def update(self, values):
        if not len(values):
            return
        count = len(values)
        mean = values.mean(axis=0)
        delta = mean - self.mean
        total = self.count + count
        self.m2 += ((values - mean) ** 2).sum(axis=0) + delta**2 * self.count * count / total
        self.mean += delta * count / total
        self.count = total
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))

    def describe(self, dimension):
        if not self.count:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        return {
            "count": self.count,
            "mean": float(self.mean[dimension]),
            "std": float(np.sqrt(self.m2[dimension] / self.count)),
            "min": float(self.minimum[dimension]),
            "max": float(self.maximum[dimension]),
        }


class QuantileSample:
    """Uniform frame sample from the smallest independent random priorities."""

    def __init__(self, size, seed):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self._keys = np.empty(2 * size)
        self._values = np.empty((2 * size, 2))
        self.count = 0
        self.threshold = 1.0

    @property
    def keys(self):
        return self._keys[: self.count]

    @property
    def values(self):
        return self._values[: self.count]

    def update(self, values):
        keys = self.rng.random(len(values))
        keep = keys < self.threshold
        keys, values = keys[keep], values[keep]
        offset = 0
        while offset < len(keys):
            count = min(len(keys) - offset, len(self._keys) - self.count)
            self._keys[self.count : self.count + count] = keys[offset : offset + count]
            self._values[self.count : self.count + count] = values[offset : offset + count]
            self.count += count
            offset += count
            if self.count == len(self._keys):
                self.trim()

    def trim(self):
        if self.count > self.size:
            chosen = np.argpartition(self.keys, self.size - 1)[: self.size]
            self._keys[: self.size] = self._keys[chosen]
            self._values[: self.size] = self._values[chosen]
            self.count = self.size
        if self.count == self.size:
            self.threshold = self.keys.max()


class FeatureStats:
    def __init__(self, target, sample_size, seed):
        self.target = target
        self.rows = self.finite = self.zeros = self.valid = 0
        self.moments = Moments()
        self.sample = QuantileSample(sample_size, seed)
        self.speakers = defaultdict(Moments)

    def update(self, raw, speaker):
        raw = torch.as_tensor(np.asarray(raw).copy(), dtype=torch.float32)
        transformed, mask = self.target.apply(raw)
        self.rows += len(raw)
        self.finite += int(torch.isfinite(raw).sum())
        self.zeros += int((raw == 0).sum())
        if not torch.isfinite(transformed[mask]).all():
            raise ValueError(f"Non-finite transformed values for {self.target.name}")
        values = np.column_stack((raw[mask].numpy(), transformed[mask].numpy())).astype(np.float64)
        self.valid += len(values)
        self.moments.update(values)
        self.sample.update(values)
        if speaker is not None:
            self.speakers[speaker].update(values)

    def report(self):
        self.sample.trim()
        percentiles = [1, 5, 25, 50, 75, 95, 99]
        result = {
            "source": self.target.source,
            "rows": self.rows,
            "finite": self.finite,
            "nonfinite": self.rows - self.finite,
            "zeros": self.zeros,
            "valid": self.valid,
            "valid_percent": 100 * self.valid / self.rows if self.rows else None,
            "zero_percent_of_finite": 100 * self.zeros / self.finite if self.finite else None,
            "quantile_sample_count": len(self.sample.values),
            "quantiles_exact": len(self.sample.values) == self.valid,
        }
        for dim, scale in enumerate(("raw_valid", "probe_target")):
            stats = self.moments.describe(dim)
            stats["percentiles"] = (
                dict(
                    zip(
                        map(str, percentiles),
                        np.percentile(self.sample.values[:, dim], percentiles).tolist(),
                        strict=True,
                    )
                )
                if self.valid
                else {}
            )
            stats["iqr"] = (
                stats["percentiles"]["75"] - stats["percentiles"]["25"] if self.valid else None
            )
            result[scale] = stats
        groups = {key: value for key, value in self.speakers.items() if value.count}
        result["speakers_with_valid_frames"] = len(groups)
        result["speaker_variation"] = {}
        for dim, scale in enumerate(("raw_valid", "probe_target")):
            if not groups:
                continue
            counts = np.array([group.count for group in groups.values()])
            means = np.array([group.mean[dim] for group in groups.values()])
            within = np.array([group.m2[dim] / group.count for group in groups.values()])
            result["speaker_variation"][scale] = {
                "frame_weighted_within_variance": float(np.average(within, weights=counts)),
                "frame_weighted_between_variance": float(
                    np.average((means - self.moments.mean[dim]) ** 2, weights=counts)
                ),
                "speaker_equal_mean": float(means.mean()),
                "speaker_equal_within_variance": float(within.mean()),
                "speaker_equal_between_variance": float(means.var()),
                "speaker_equal_std": float(np.sqrt(within.mean() + means.var())),
                "speaker_mean_percentiles": dict(
                    zip(
                        map(str, percentiles),
                        np.percentile(means, percentiles).tolist(),
                        strict=True,
                    )
                ),
            }
        result["per_speaker"] = {
            key: {"raw_valid": group.describe(0), "probe_target": group.describe(1)}
            for key, group in groups.items()
        }
        return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", nargs="+")
    grouping = parser.add_mutually_exclusive_group()
    grouping.add_argument(
        "--dataset-config", type=Path, help="Trusted YAML with dataset node; reads metadata only."
    )
    grouping.add_argument(
        "--speaker-regex",
        help="Regex with one capture group, matched against relative feature IDs.",
    )
    parser.add_argument("--speaker-resource", default="spkid")
    defaults = [
        name
        for name, target in TARGETS.items()
        if isinstance(target.task, RegressionTask) and not target.resources
    ]
    parser.add_argument("--targets", nargs="+", choices=defaults, default=defaults)
    parser.add_argument("--quantile-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, help="First N files for a smoke test, not a representative subset."
    )
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("Seed must be nonnegative")
    if args.quantile_samples <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("Sample size and limit must be positive")
    if args.dataset_config and args.splits:
        parser.error("Select partitions in the dataset config rather than --splits")
    if len(args.targets) != len(set(args.targets)):
        parser.error("Targets must be distinct")
    if args.speaker_regex:
        try:
            args.speaker_regex = re.compile(args.speaker_regex)
        except re.error as error:
            parser.error(str(error))
        if args.speaker_regex.groups != 1:
            parser.error("--speaker-regex must contain exactly one capture group")
    return args


def main(argv=None):
    args = parse_args(argv)
    root = args.feature_root
    if not root.is_dir():
        raise ValueError(f"Feature root does not exist: {root}")
    roots = [root / split for split in args.splits] if args.splits else [root]
    for directory in roots:
        if not directory.is_dir():
            raise ValueError(f"Feature directory does not exist: {directory}")
    files = sorted({path for directory in roots for path in directory.rglob("*.parquet")})
    speaker_map = None
    missing = []
    if args.dataset_config:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        print(
            "Loading dataset metadata; configured boundary probes may take several minutes.",
            flush=True,
        )
        dataset = instantiate(OmegaConf.load(args.dataset_config).dataset, load=[])
        speaker_map = {}
        for row in dataset.rows:
            key = str(utterance_relative_path(row.utt_id, row.split))
            if key in speaker_map:
                raise ValueError(f"Duplicate feature ID in config: {key}")
            speaker = row.resources[args.speaker_resource].value
            if not isinstance(speaker, str) or not speaker:
                raise ValueError(f"Missing speaker identity for {key}")
            speaker_map[key] = speaker
        available = {str(path.relative_to(root).with_suffix("")): path for path in files}
        missing = sorted(speaker_map.keys() - available.keys())
        files = sorted(available[key] for key in speaker_map.keys() & available.keys())
    if not files:
        raise ValueError("No matching Parquet files")
    available_count = len(files)
    files = files[: args.limit] if args.limit is not None else files
    targets = [TARGETS[name] for name in args.targets]
    columns = list(dict.fromkeys(target.source for target in targets))
    stats = {
        target.name: FeatureStats(target, args.quantile_samples, args.seed + index)
        for index, target in enumerate(targets)
    }
    empty, all_missing, failures = [], [], []
    metadata = [
        {"path": str(path.relative_to(root)), "settings": json.loads(path.read_text())}
        for directory in roots
        for path in sorted(directory.rglob("_metadata.json"))
    ]
    # Ignore split labels when checking that this store used consistent extraction settings.
    settings = {
        json.dumps(
            {key: value for key, value in item["settings"].items() if key != "split"},
            sort_keys=True,
        )
        for item in metadata
    }
    if len(settings) > 1:
        raise ValueError("Extraction metadata differs within this store; analyze separately")
    for path in tqdm(files, desc="Feature distributions"):
        key = str(path.relative_to(root).with_suffix(""))
        speaker = speaker_map[key] if speaker_map is not None else None
        if args.speaker_regex:
            match = args.speaker_regex.search(key)
            if not match:
                raise ValueError(f"Speaker regex did not match {key!r}")
            speaker = match.group(1)
        try:
            frame = pd.read_parquet(path, columns=columns)
            if frame.empty:
                empty.append(key)
                continue
            if not np.isfinite(frame.to_numpy(dtype=float)).any():
                all_missing.append(key)
        except Exception as error:
            failures.append({"id": key, "error": str(error)})
            continue
        for target in targets:
            stats[target.name].update(frame[target.source].to_numpy(), speaker)
    report = {
        "schema_version": 1,
        "feature_root": str(root),
        "splits": args.splits,
        "available_files": available_count,
        "scanned_files": len(files),
        "limited": len(files) < available_count,
        "missing_feature_ids": missing,
        "empty_files": empty,
        "all_selected_features_nonfinite": all_missing,
        "read_errors": failures,
        "extraction_metadata": metadata,
        "grouping": "dataset_resource"
        if speaker_map is not None
        else "regex"
        if args.speaker_regex
        else None,
        "quantile_sample_size": args.quantile_samples,
        "seed": args.seed,
        "notes": [
            "Population standard deviations; valid masks and transforms match FrameTarget.apply at source feature rate.",
            "Percentiles use a uniform frame sample; counts and moments use every valid frame.",
            "Speaker-equal moments give each speaker equal weight; pooled percentiles remain frame-weighted.",
            "Finite/mask-valid does not establish physiological reliability or independent voicing confidence.",
        ],
        "features": {name: value.report() for name, value in stats.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"Scanned {len(files)} files; empty {len(empty)}; all nonfinite {len(all_missing)}; read errors {len(failures)}"
    )
    for name, item in report["features"].items():
        value = item["probe_target"]
        print(f"{name}: valid={item['valid_percent']}, mean={value['mean']}, std={value['std']}")
    print(f"Saved: {args.out}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
