from __future__ import annotations

import math
import os

import pytest
import torch
from quick_convert.components.ssl import S3TokenizerContentEncoder

from ssl_probe.dataset import FrameDataset
from ssl_probe.encoders import build_layer_fusion
from ssl_probe.probe import Probe

pytestmark = pytest.mark.model

if os.environ.get("SSL_PROBE_RUN_MODEL_TESTS") != "1":
    pytest.skip(
        "Set SSL_PROBE_RUN_MODEL_TESTS=1 to run model-backed smoke tests.",
        allow_module_level=True,
    )


def test_s3tokenizer_layers_flow_through_trainable_fusion() -> None:
    sample_rate = 16_000
    time = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
    waveform = (0.01 * torch.sin(2 * math.pi * 220 * time)).unsqueeze(0)
    lengths = torch.tensor([waveform.shape[-1]], dtype=torch.long)

    encoder = S3TokenizerContentEncoder(
        representation="encoder",
        layer=-1,
        device="cpu",
    )
    features = encoder.encode_waveforms(
        waveform,
        lengths=lengths,
        sample_rate=sample_rate,
    )

    assert features.values.ndim == 4
    assert features.values.shape[0] == 1
    assert features.values.shape[2:] == (12, 1280)
    assert features.lengths.shape == (1,)
    assert 0 < int(features.lengths[0]) <= features.values.shape[1]
    assert features.feature_dim == 1280
    assert features.frame_hz == 25.0
    assert features.layer == "all"

    utterance = FrameDataset._normalize_content_frames(
        features.values[0, : int(features.lengths[0])],
        preserve_layers=True,
    )
    assert utterance.shape[1:] == (12, 1280)

    fusion = build_layer_fusion("weighted-sum", encoder)
    assert fusion is not None
    probe = Probe(
        input_dim=1280,
        output_dim=1,
        feature_transform=fusion,
    )

    probe_input = utterance[:1].unsqueeze(0).detach().clone()
    prediction = probe(probe_input)
    prediction.square().mean().backward()

    assert prediction.shape == (1, 1)
    assert torch.isfinite(prediction).all()
    assert fusion.weights.grad is not None
    assert torch.isfinite(fusion.weights.grad).all()
