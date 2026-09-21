import pytest
import torch

from ssl_probe.dataset import FrameDataset


def test_align_target_frames_to_encoder_length() -> None:
    target = torch.arange(10)

    aligned = FrameDataset._align_target_frames(target, output_frames=5)

    assert torch.equal(aligned, torch.tensor([0, 2, 4, 6, 8]))


def test_normalize_singleton_layer_axis() -> None:
    values = torch.zeros(5, 1, 3)

    normalized = FrameDataset._normalize_content_frames(values)

    assert normalized.shape == (5, 3)


def test_reject_multiple_vectors_per_frame() -> None:
    values = torch.zeros(5, 12, 3)

    with pytest.raises(ValueError, match="select a single layer"):
        FrameDataset._normalize_content_frames(values)
