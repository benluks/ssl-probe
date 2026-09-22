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
        self.sample_rate = 16_000
        self.layer = 4

    def encode_file(self, path: str | Path) -> ContentFeatures:
        raise NotImplementedError

    def encode_waveforms(
        self,
        wavs: torch.FloatTensor,
        lengths: torch.LongTensor | None = None,
        sample_rates: torch.LongTensor | None = None,
    ) -> ContentFeatures:
        raise NotImplementedError


def test_resolve_content_encoder_alias() -> None:
    assert resolve_content_encoder_class("wavlm") is WavLMContentEncoder
    assert resolve_content_encoder_class("s3tokenizer") is S3TokenizerContentEncoder


def test_content_encoder_slug() -> None:
    assert content_encoder_slug(ExampleContentEncoder()) == "example-l4"
