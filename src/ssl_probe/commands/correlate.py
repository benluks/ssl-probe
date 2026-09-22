from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class CorrelationAccumulator:
    feature_names: list[str]

    def __post_init__(self):
        n_features = len(self.feature_names)

        self.n = np.zeros(n_features, dtype=np.int64)
        self.sum_x = np.zeros(n_features, dtype=np.float64)
        self.sum_y = np.zeros(n_features, dtype=np.float64)
        self.sum_x2 = np.zeros(n_features, dtype=np.float64)
        self.sum_y2 = np.zeros(n_features, dtype=np.float64)
        self.sum_xy = np.zeros(n_features, dtype=np.float64)

    def update(
        self,
        x: np.ndarray,
        y: np.ndarray,
        valid: np.ndarray | None = None,
    ) -> None:
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape=} vs {y.shape=}")

        finite = np.isfinite(x) & np.isfinite(y)

        if valid is None:
            valid = finite
        else:
            valid = valid & finite

        x_valid = np.where(valid, x, 0.0)
        y_valid = np.where(valid, y, 0.0)

        self.n += valid.sum(axis=0)
        self.sum_x += x_valid.sum(axis=0)
        self.sum_y += y_valid.sum(axis=0)
        self.sum_x2 += (x_valid * x_valid).sum(axis=0)
        self.sum_y2 += (y_valid * y_valid).sum(axis=0)
        self.sum_xy += (x_valid * y_valid).sum(axis=0)

    def compute(self) -> pd.DataFrame:
        n = self.n.astype(np.float64)

        pearson = np.full(len(n), np.nan, dtype=np.float64)
        ccc = np.full(len(n), np.nan, dtype=np.float64)

        usable = self.n >= 2

        # Means
        mean_x = np.full_like(n, np.nan)
        mean_y = np.full_like(n, np.nan)

        mean_x[usable] = self.sum_x[usable] / n[usable]
        mean_y[usable] = self.sum_y[usable] / n[usable]

        # Population variances and covariance
        var_x = np.full_like(n, np.nan)
        var_y = np.full_like(n, np.nan)
        cov_xy = np.full_like(n, np.nan)

        var_x[usable] = self.sum_x2[usable] / n[usable] - mean_x[usable] ** 2
        var_y[usable] = self.sum_y2[usable] / n[usable] - mean_y[usable] ** 2
        cov_xy[usable] = self.sum_xy[usable] / n[usable] - mean_x[usable] * mean_y[usable]

        # Guard against tiny negative values from floating-point error.
        var_x = np.maximum(var_x, 0.0)
        var_y = np.maximum(var_y, 0.0)

        # Pearson
        pearson_denominator = np.sqrt(var_x * var_y)
        pearson_usable = usable & (pearson_denominator > 0)

        pearson[pearson_usable] = cov_xy[pearson_usable] / pearson_denominator[pearson_usable]

        # Concordance correlation coefficient
        ccc_denominator = var_x + var_y + (mean_x - mean_y) ** 2

        ccc_usable = usable & (ccc_denominator > 0)

        ccc[ccc_usable] = 2.0 * cov_xy[ccc_usable] / ccc_denominator[ccc_usable]

        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "pearson_r": pearson,
                "ccc": ccc,
                "mean_original": mean_x,
                "mean_converted": mean_y,
                "std_original": np.sqrt(var_x),
                "std_converted": np.sqrt(var_y),
                "n": self.n,
            }
        )


def get_valid_mask(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    """
    Build a [T, F] validity mask.

    By default, all finite paired values are valid.

    Add feature-specific rules here where openSMILE uses a sentinel value
    rather than NaN for undefined measurements.
    """
    valid = np.isfinite(x) & np.isfinite(y)

    for i, feature in enumerate(feature_names):
        name = feature.lower()

        # eGeMAPS F0 LLDs use 0 for unvoiced / undefined F0.
        #
        # This catches names such as:
        # F0semitoneFrom27.5Hz_sma3nz
        if "f0" in name:
            valid[:, i] &= (x[:, i] > 0) & (y[:, i] > 0)

    return valid


def load_feature_pair(
    original_path: Path,
    converted_path: Path,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    original_df = pd.read_parquet(original_path)
    converted_df = pd.read_parquet(converted_path)

    if feature_names is None:
        # Only compare columns that exist in both files and are numeric.
        common = [
            column
            for column in original_df.columns
            if column in converted_df.columns
            and pd.api.types.is_numeric_dtype(original_df[column])
            and pd.api.types.is_numeric_dtype(converted_df[column])
        ]

        if not common:
            raise ValueError(
                f"No common numeric feature columns in:\n  {original_path}\n  {converted_path}"
            )

        feature_names = common

    missing_original = set(feature_names) - set(original_df.columns)
    missing_converted = set(feature_names) - set(converted_df.columns)

    if missing_original:
        raise ValueError(f"{original_path} is missing features: {sorted(missing_original)}")

    if missing_converted:
        raise ValueError(f"{converted_path} is missing features: {sorted(missing_converted)}")

    original = original_df[feature_names].to_numpy(dtype=np.float64)
    converted = converted_df[feature_names].to_numpy(dtype=np.float64)

    # Conversion/resynthesis can occasionally produce an off-by-one frame
    # difference. Compare only the overlapping portion.
    length = min(len(original), len(converted))

    return (
        original[:length],
        converted[:length],
        feature_names,
    )


def compute_split_correlations(
    original_dir: Path,
    converted_dir: Path,
) -> pd.DataFrame:
    original_files = {path.stem: path for path in original_dir.glob("*.parquet")}
    converted_files = {path.stem: path for path in converted_dir.glob("*.parquet")}

    common_utt_ids = sorted(original_files.keys() & converted_files.keys())

    if not common_utt_ids:
        raise RuntimeError(
            f"No matching parquet files between:\n  {original_dir}\n  {converted_dir}"
        )

    missing_converted = original_files.keys() - converted_files.keys()
    missing_original = converted_files.keys() - original_files.keys()

    if missing_converted:
        print(f"Warning: {len(missing_converted)} original utterances have no converted features.")

    if missing_original:
        print(f"Warning: {len(missing_original)} converted utterances have no original features.")

    print(f"Matched utterances: {len(common_utt_ids)}")

    accumulator: CorrelationAccumulator | None = None
    feature_names: list[str] | None = None

    total_frames = 0

    for index, utt_id in tqdm(enumerate(common_utt_ids, start=1), total=len(common_utt_ids)):
        original, converted, current_features = load_feature_pair(
            original_files[utt_id],
            converted_files[utt_id],
            feature_names=feature_names,
        )

        if feature_names is None:
            feature_names = current_features
            accumulator = CorrelationAccumulator(feature_names)

            print(f"Features: {len(feature_names)}")

        assert accumulator is not None
        assert feature_names is not None

        valid = get_valid_mask(
            original,
            converted,
            feature_names,
        )

        accumulator.update(
            original,
            converted,
            valid=valid,
        )

        total_frames += len(original)

        if index % 1000 == 0 or index == len(common_utt_ids):
            print(f"{index:>7}/{len(common_utt_ids)} utterances | {total_frames:,} aligned frames")

    assert accumulator is not None

    results = accumulator.compute()

    # Useful ordering for inspection.
    results = results.sort_values(
        "pearson_r",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "split",
        help="Dataset split, e.g. test-other",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("features/statistics"),
    )
    parser.add_argument(
        "--original",
        default="librispeech",
        help="Directory containing original openSMILE features.",
    )
    parser.add_argument(
        "--converted",
        default="knnvc_original_librispeech",
        help="Directory containing converted openSMILE features.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    original_dir = args.feature_root / args.original / args.split
    converted_dir = args.feature_root / args.converted / args.split

    results = compute_split_correlations(
        original_dir=original_dir,
        converted_dir=converted_dir,
    )

    print()
    print(results.to_string(index=False))

    valid_r = results["pearson_r"].dropna()
    valid_ccc = results["ccc"].dropna()

    print()
    print(f"Mean feature Pearson r:    {valid_r.mean():.4f}")
    print(f"Median feature Pearson r:  {valid_r.median():.4f}")
    print(f"Mean feature CCC:          {valid_ccc.mean():.4f}")
    print(f"Median feature CCC:        {valid_ccc.median():.4f}")
    print(f"Features included:         {len(valid_ccc)}/{len(results)}")

    output = args.output
    if output is None:
        output = args.feature_root / f"{args.converted}_vs_{args.original}" / f"{args.split}.csv"

    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
