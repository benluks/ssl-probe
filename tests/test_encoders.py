from pathlib import Path

import torch
from quick_convert.components.ssl import (
    ContentEncoder,
    ContentFeatures,
    S3TokenizerContentEncoder,
    WavLMContentEncoder,
)

from ssl_probe.encoders import content_encoder_slug, resolve_content_encoder_class


class ExampleContentEncoder(ContentEncoder):
    FEATURE_DIM = 3

    def __init__(self) -> None:
        super().__init__(device="cpu")
        self.layer = 4

    @property
    def sample_rate(self) -> int:
        return 16_000

    @property
    def frame_hz(self) -> float:
        return 50.0

    def forward(self, batch, **kwargs) -> ContentFeatures:
        raise NotImplementedError

    def encode_file(self, path: str | Path) -> ContentFeatures:
        raise NotImplementedError

    def encode_waveforms(
        self,
        waveforms: torch.FloatTensor,
        lengths: torch.LongTensor | None = None,
        sample_rate: int | None = None,
    ) -> ContentFeatures:
        raise NotImplementedError


def test_resolve_content_encoder_alias() -> None:
    assert resolve_content_encoder_class("wavlm") is WavLMContentEncoder
    assert resolve_content_encoder_class("s3tokenizer") is S3TokenizerContentEncoder


def test_content_encoder_slug() -> None:
    assert content_encoder_slug(ExampleContentEncoder()) == "example-l4"
