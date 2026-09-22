from pathlib import Path

import pytest
import torch
from quick_convert.components.layers import LayerWeightedSum
from quick_convert.components.ssl import (
    ContentEncoder,
    ContentFeatures,
    S3TokenizerContentEncoder,
    WavLMContentEncoder,
)

from ssl_probe.encoders import (
    build_layer_fusion,
    content_encoder_num_layers,
    content_encoder_slug,
    resolve_content_encoder_class,
)


class ExampleContentEncoder(ContentEncoder):
    FEATURE_DIM = 3
    N_LAYERS = 12

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


def test_content_encoder_layer_count() -> None:
    assert content_encoder_num_layers(ExampleContentEncoder()) == 12


def test_build_weighted_sum_layer_fusion() -> None:
    encoder = ExampleContentEncoder()

    fusion = build_layer_fusion("weighted-sum", encoder)

    assert isinstance(fusion, LayerWeightedSum)
    assert fusion.weights.shape == (1, 12)
    assert content_encoder_slug(encoder, fusion) == "example-l4-wsum12"


def test_layer_count_override_supports_arbitrary_encoders() -> None:
    fusion = build_layer_fusion(
        "weighted-sum",
        ExampleContentEncoder(),
        num_layers=7,
    )

    assert fusion is not None
    assert fusion.weights.shape == (1, 7)


def test_num_layers_requires_layer_fusion() -> None:
    with pytest.raises(ValueError, match="requires --layer-fusion"):
        build_layer_fusion("none", ExampleContentEncoder(), num_layers=12)
