import json

import pytest
import torch

from ssl_probe.dataset import FrameDataset
from ssl_probe.targets import TARGETS, SmileParquetStore


class FakeContentEncoder:
    sample_rate = 16_000
    frame_hz = 50.0
    feature_dim = 8

    def eval(self):
        return self


def test_align_target_frames_to_encoder_length() -> None:
    target = torch.arange(10)

    aligned = FrameDataset._align_target_frames(
        target,
        output_frames=5,
        target_frame_hz=100.0,
        content_frame_hz=50.0,
    )

    assert torch.equal(aligned, torch.tensor([0, 2, 4, 6, 8]))


def test_long_wavlm_alignment_exactly_matches_legacy_stride():
    target = torch.arange(20000)
    aligned = FrameDataset._align_target_frames(
        target, 10000, target_frame_hz=100.0, content_frame_hz=50.0
    )
    assert torch.equal(aligned, target[::2])


def test_fractional_alignment_preserves_floor_and_integer_boundaries():
    target = torch.arange(2000)
    aligned = FrameDataset._align_target_frames(
        target, 1000, target_frame_hz=100.0, content_frame_hz=60.0
    )
    assert torch.equal(aligned, torch.arange(1000) * 5 // 3)


def test_normalize_singleton_layer_axis() -> None:
    values = torch.zeros(5, 1, 3)

    normalized = FrameDataset._normalize_content_frames(values)

    assert normalized.shape == (5, 3)


def test_reject_multiple_vectors_per_frame() -> None:
    values = torch.zeros(5, 12, 3)

    with pytest.raises(ValueError, match="select a single layer"):
        FrameDataset._normalize_content_frames(values)


def test_preserve_multiple_layer_vectors_per_frame() -> None:
    values = torch.zeros(5, 12, 3)

    normalized = FrameDataset._normalize_content_frames(
        values,
        preserve_layers=True,
    )

    assert normalized.shape == (5, 12, 3)


def test_layer_fusion_rejects_single_layer_content() -> None:
    values = torch.zeros(5, 3)

    with pytest.raises(ValueError, match="Layer fusion requires"):
        FrameDataset._normalize_content_frames(
            values,
            preserve_layers=True,
        )


def test_alignment_uses_timebase_instead_of_sequence_proportion() -> None:
    target = torch.arange(10)

    aligned = FrameDataset._align_target_frames(
        target,
        output_frames=3,
        target_frame_hz=100.0,
        content_frame_hz=100.0,
    )

    assert torch.equal(aligned, torch.tensor([0, 1, 2]))


def test_alignment_supports_25_hz_content() -> None:
    target = torch.arange(10)

    aligned = FrameDataset._align_target_frames(
        target,
        output_frames=3,
        target_frame_hz=100.0,
        content_frame_hz=25.0,
    )

    assert torch.equal(aligned, torch.tensor([0, 4, 8]))


def test_alignment_truncates_frames_beyond_target_duration() -> None:
    target = torch.arange(10)

    aligned = FrameDataset._align_target_frames(
        target,
        output_frames=6,
        target_frame_hz=100.0,
        content_frame_hz=50.0,
    )

    assert torch.equal(aligned, torch.tensor([0, 2, 4, 6, 8]))


def test_alignment_rejects_missing_timebase() -> None:
    with pytest.raises(ValueError, match="frame rates must be positive"):
        FrameDataset._align_target_frames(
            torch.arange(10),
            output_frames=5,
            target_frame_hz=100.0,
            content_frame_hz=0.0,
        )


def test_smile_store_reads_frame_rate_metadata(tmp_path) -> None:
    (tmp_path / "_metadata.json").write_text(json.dumps({"frame_hz": 80.0}))

    store = SmileParquetStore(tmp_path, feature_column="feature")

    assert store.frame_hz == 80.0


def test_smile_store_defaults_existing_outputs_to_100_hz(tmp_path) -> None:
    store = SmileParquetStore(tmp_path, feature_column="feature")

    assert store.frame_hz == 100.0


def test_frame_dataset_accepts_split_qualified_manifest_ids(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "train.csv"
    manifest.write_text(
        "utt_id,path,split,spkid\n"
        "picnic/1234,/audio/picnic/1234.wav,picnic,1234\n"
        "rainbow/5678,/audio/rainbow/5678.wav,rainbow,5678\n"
    )
    smile_root = tmp_path / "features" / "opensmile" / "clac"
    monkeypatch.setattr(FrameDataset, "_estimate_reference_frames", lambda self, path: 10)

    dataset = FrameDataset(
        content_encoder=FakeContentEncoder(),
        target=TARGETS["spectral_flux"],
        manifest_path=manifest,
        smile_root=smile_root,
    )

    assert [row.utt_id for row in dataset.base_dataset.rows] == [
        "picnic/1234",
        "rainbow/5678",
    ]
    assert [row.split for row in dataset.base_dataset.rows] == ["picnic", "rainbow"]
    assert dataset.smile_features.smile_root == smile_root


def test_multiple_split_smile_stores(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(FrameDataset, "_estimate_reference_frames", lambda self, path: 10)
    rows = [
        SimpleNamespace(utt_id="a", path="/unused/a.wav", split="train-clean-100"),
        SimpleNamespace(utt_id="b", path="/unused/b.wav", split="train-clean-360"),
    ]

    class FakeDataset:
        def __init__(self, rows):
            self.rows = rows

    monkeypatch.setattr(
        "ssl_probe.dataset.load_dataset",
        lambda *args, **kwargs: FakeDataset(rows),
    )
    root = tmp_path / "smile"
    for split in ("train-clean-100", "train-clean-360"):
        (root / split).mkdir(parents=True)
        (root / split / "_metadata.json").write_text(json.dumps({"frame_hz": 100.0}))

    dataset = FrameDataset(
        content_encoder=FakeContentEncoder(),
        target=TARGETS["spectral_flux"],
        root=tmp_path,
        splits=["train-clean-100", "train-clean-360"],
        smile_root=root,
        split_smile_roots={split: root / split for split in ("train-clean-100", "train-clean-360")},
    )
    assert dataset._smile_store(rows[0]).smile_root == root / "train-clean-100"
    assert dataset._smile_store(rows[1]).smile_root == root / "train-clean-360"
