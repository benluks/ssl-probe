import json

import pytest
import torch
from ssl_probe.targets import TARGETS, SmileParquetStore

from ssl_probe.dataset import FrameDataset


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
