import json

import numpy as np
import pandas as pd
import pytest

from ssl_probe.commands.feature_stats import FeatureStats, QuantileSample, main, parse_args
from ssl_probe.targets import TARGETS


def test_registry_masks_transforms_and_zero_accounting():
    stats = FeatureStats(TARGETS["jitter"], sample_size=100, seed=0)
    stats.update(np.array([0, 0.1, 0.3, np.nan]), "a")
    report = stats.report()
    assert (report["rows"], report["finite"], report["zeros"], report["valid"]) == (4, 3, 1, 2)
    assert report["raw_valid"]["mean"] == pytest.approx(0.2)
    assert report["probe_target"]["mean"] == pytest.approx(np.log1p([0.1, 0.3]).mean())
    assert report["quantiles_exact"]
    hnr = FeatureStats(TARGETS["hnr"], sample_size=100, seed=0)
    hnr.update(np.array([0, -2, 6, np.nan]), "a")
    assert hnr.report()["raw_valid"]["mean"] == pytest.approx(2)
    assert hnr.report()["valid"] == 2


def test_weighted_and_speaker_equal_variance_decomposition():
    stats = FeatureStats(TARGETS["shimmer"], sample_size=100, seed=0)
    stats.update(np.array([1, 3]), "a")
    stats.update(np.array([5]), "a")
    stats.update(np.array([10]), "b")
    report = stats.report()
    values = np.array([1, 3, 5, 10])
    variation = report["speaker_variation"]["raw_valid"]
    assert report["raw_valid"]["mean"] == pytest.approx(values.mean())
    assert report["raw_valid"]["std"] ** 2 == pytest.approx(values.var())
    assert variation["frame_weighted_within_variance"] == pytest.approx(2)
    assert variation["frame_weighted_within_variance"] + variation[
        "frame_weighted_between_variance"
    ] == pytest.approx(values.var())
    assert variation["speaker_equal_mean"] == pytest.approx(6.5)
    assert variation["speaker_equal_within_variance"] == pytest.approx(4 / 3)
    assert report["per_speaker"]["a"]["raw_valid"]["count"] == 3


def test_quantile_sample_keeps_exact_smallest_priorities_across_chunks():
    sample = QuantileSample(17, 11)
    values = np.column_stack((np.arange(1000), np.arange(1000)))
    for chunk in np.array_split(values, 41):
        sample.update(chunk)
    sample.trim()
    keys = np.random.default_rng(11).random(1000)
    expected = np.argsort(keys)[:17]
    assert set(sample.values[:, 0]) == set(expected)
    assert len(sample.keys) == 17


def test_parquet_audit_ignores_timestamps_and_handles_missing_schema(tmp_path):
    root = tmp_path / "features"
    root.mkdir()
    columns = [TARGETS[key].source for key in ("jitter", "shimmer", "hnr")]
    frame = pd.DataFrame({column: [0.0, 1.0, np.nan] for column in columns})
    frame["start"] = [0, 1, 2]
    frame["end"] = [1, 2, 3]
    frame.to_parquet(root / "a-1.parquet")
    missing = pd.DataFrame({column: [np.nan] for column in columns})
    missing["start"] = [0.0]
    missing["end"] = [0.01]
    missing.to_parquet(root / "b-1.parquet")
    frame.iloc[:0].to_parquet(root / "c-1.parquet")
    pd.DataFrame({"start": [0]}).to_parquet(root / "d-1.parquet")
    output = tmp_path / "stats.json"
    with pytest.raises(SystemExit):
        main(
            [
                "--feature-root",
                str(root),
                "--out",
                str(output),
                "--targets",
                "jitter",
                "shimmer",
                "hnr",
                "--speaker-regex",
                r"^([a-z])-",
                "--quantile-samples",
                "20",
            ]
        )
    report = json.loads(output.read_text())
    assert report["scanned_files"] == 4
    assert report["all_selected_features_nonfinite"] == ["b-1"]
    assert report["empty_files"] == ["c-1"]
    assert len(report["read_errors"]) == 1
    assert report["features"]["jitter"]["rows"] == 4
    assert report["features"]["jitter"]["valid"] == 1
    assert "start" not in report["features"]


def test_dataset_grouping_uses_resources_without_audio_loading(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from hydra import utils
    from quick_convert.data import MetadataSample
    from quick_convert.data.resources import ResourceCollection, ResourceRef

    root = tmp_path / "features"
    root.mkdir()
    for name in ("a", "b", "extra"):
        pd.DataFrame({TARGETS["shimmer"].source: [1.0]}).to_parquet(root / f"{name}.parquet")
    rows = [
        MetadataSample(
            utt_id=name,
            path=tmp_path / "missing-audio.mp4",
            resources=ResourceCollection.from_refs(
                [ResourceRef(name="spkid", kind="text", value="linked")]
            ),
        )
        for name in ("a", "b", "missing")
    ]

    def metadata_only(config, **kwargs):
        assert kwargs["load"] == []
        return SimpleNamespace(rows=rows)

    monkeypatch.setattr(utils, "instantiate", metadata_only)
    config = tmp_path / "data.yaml"
    config.write_text("dataset: {}\n")
    out = tmp_path / "stats.json"
    main(
        [
            "--feature-root",
            str(root),
            "--out",
            str(out),
            "--dataset-config",
            str(config),
            "--targets",
            "shimmer",
        ]
    )
    report = json.loads(out.read_text())
    assert report["scanned_files"] == 2
    assert report["missing_feature_ids"] == ["missing"]
    assert report["features"]["shimmer"]["speakers_with_valid_frames"] == 1
    assert report["features"]["shimmer"]["per_speaker"]["linked"]["raw_valid"]["count"] == 2


@pytest.mark.parametrize(
    "extra", [["--quantile-samples", "0"], ["--limit", "-1"], ["--speaker-regex", "abc"]]
)
def test_invalid_sampling_and_grouping_arguments(extra):
    with pytest.raises(SystemExit):
        parse_args(["--feature-root", "features", "--out", "stats.json", *extra])
