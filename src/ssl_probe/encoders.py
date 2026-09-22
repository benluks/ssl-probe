from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from quick_convert.components.layers import LayerWeightedSum
from quick_convert.components.ssl import ContentEncoder
from quick_convert.components.ssl import (
    resolve_content_encoder as resolve_content_encoder_class,
)

ENCODER_DEFAULTS: dict[str, dict[str, Any]] = {
    "wavlm": {"layer": 6},
}


def build_content_encoder(
    name: str,
    kwargs: Mapping[str, Any] | None = None,
) -> ContentEncoder:
    encoder_kwargs = dict(ENCODER_DEFAULTS.get(name, {}))
    encoder_kwargs.update(kwargs or {})
    return resolve_content_encoder_class(name)(**encoder_kwargs)


def content_encoder_num_layers(encoder: ContentEncoder) -> int | None:
    num_layers = getattr(encoder, "N_LAYERS", None)

    if num_layers is None:
        model = getattr(encoder, "model", None)
        config = getattr(model, "config", None)
        for attribute in ("num_hidden_layers", "num_layers", "encoder_layers"):
            num_layers = getattr(config, attribute, None)
            if num_layers is not None:
                break

    if not isinstance(num_layers, int) or num_layers <= 0:
        return None

    downsample_factor = getattr(encoder, "downsample_factor", 0)
    if isinstance(downsample_factor, int) and downsample_factor > 1:
        num_layers //= downsample_factor

    return num_layers or None


def build_layer_fusion(
    mode: str,
    encoder: ContentEncoder,
    num_layers: int | None = None,
) -> LayerWeightedSum | None:
    if mode == "none":
        if num_layers is not None:
            raise ValueError("--num-layers requires --layer-fusion weighted-sum.")
        return None

    if mode != "weighted-sum":
        raise ValueError(f"Unknown layer fusion mode: {mode!r}.")

    layer = getattr(encoder, "layer", None)
    if layer not in (None, -1):
        raise ValueError(
            "Layer fusion requires the encoder to return all layers; "
            "configure its layer argument accordingly."
        )

    representation = getattr(encoder, "representation", None)
    if representation not in (None, "encoder"):
        raise ValueError(
            "Layer fusion is available only for multi-layer encoder representations, "
            f"not representation={representation!r}."
        )

    if num_layers is not None and num_layers <= 0:
        raise ValueError("The number of representation layers must be positive.")

    resolved_num_layers = (
        num_layers if num_layers is not None else content_encoder_num_layers(encoder)
    )
    if resolved_num_layers is None:
        raise ValueError(
            f"Cannot infer the number of output layers from {type(encoder).__name__}; "
            "pass --num-layers explicitly."
        )

    return LayerWeightedSum(num_layers=resolved_num_layers)


def content_encoder_slug(
    encoder: ContentEncoder,
    layer_fusion: LayerWeightedSum | None = None,
) -> str:
    name = type(encoder).__name__.removesuffix("ContentEncoder")
    name = re.sub(r"(?<!^)(?=[A-Z])", "-", name).lower()
    layer = getattr(encoder, "layer", None)
    slug = f"{name}-l{layer}" if layer is not None else name

    if layer_fusion is not None:
        slug = f"{slug}-wsum{layer_fusion.weights.shape[1]}"

    return slug
