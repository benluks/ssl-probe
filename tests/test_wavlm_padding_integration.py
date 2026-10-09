"""Opt-in real WavLM padding invariance check.

Run with: SSL_PROBE_TEST_WAVLM_PADDING=1 uv run pytest -s tests/test_wavlm_padding_integration.py
Requires access to the WavLM checkpoint (may download it).
"""

import os

import pytest
import torch
from quick_convert.data import AudioBatch, AudioSample

from ssl_probe.encoders import build_content_encoder


@pytest.mark.skipif(
    os.getenv("SSL_PROBE_TEST_WAVLM_PADDING") != "1",
    reason="Opt in to downloading/running the real WavLM checkpoint.",
)
def test_wavlm_layer6_is_padding_invariant():
    torch.manual_seed(115)
    encoder = build_content_encoder("wavlm", {"layer": 6})
    encoder.eval()
    short = AudioSample(
        utt_id="short",
        path="short.wav",
        waveform=torch.randn(1, 16000),
        sample_rate=16000,
    )
    long = AudioSample(
        utt_id="long",
        path="long.wav",
        waveform=torch.randn(1, 32000),
        sample_rate=16000,
    )
    with torch.inference_mode():
        alone = encoder(AudioBatch.from_samples([short]))
        padded = encoder(AudioBatch.from_samples([short, long]))
    count = int(alone.lengths[0])
    assert int(padded.lengths[0]) == count
    delta = (alone.values[0, :count] - padded.values[0, :count]).abs()
    print(f"WavLM padding delta: max={delta.max().item():.6g}, mean={delta.mean().item():.6g}")
    torch.testing.assert_close(
        alone.values[0, :count],
        padded.values[0, :count],
        atol=1e-4,
        rtol=1e-4,
    )
