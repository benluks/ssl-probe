from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

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


def content_encoder_slug(encoder: ContentEncoder) -> str:
    name = type(encoder).__name__.removesuffix("ContentEncoder")
    name = re.sub(r"(?<!^)(?=[A-Z])", "-", name).lower()
    layer = getattr(encoder, "layer", None)
    return f"{name}-l{layer}" if layer is not None else name
