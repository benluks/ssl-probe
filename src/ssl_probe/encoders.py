from __future__ import annotations

import importlib
import re
from collections.abc import Mapping
from typing import Any

from quick_convert.components.ssl import ContentEncoder

ENCODER_ALIASES = {
    "dac": "quick_convert.components.ssl.DACContentEncoder",
    "w2vbert": "quick_convert.components.ssl.W2VBertContentEncoder",
    "wavlm": "quick_convert.components.ssl.WavLMContentEncoder",
}

ENCODER_DEFAULTS: dict[str, dict[str, Any]] = {
    "wavlm": {"layer": 6},
}


def resolve_content_encoder_class(name: str) -> type[ContentEncoder]:
    dotted_path = ENCODER_ALIASES.get(name, name)
    module_name, separator, class_name = dotted_path.rpartition(".")
    if not separator:
        aliases = ", ".join(sorted(ENCODER_ALIASES))
        raise ValueError(
            f"Unknown content encoder {name!r}. Use one of {aliases}, or a dotted class path."
        )

    module = importlib.import_module(module_name)
    encoder_class = getattr(module, class_name)
    if not isinstance(encoder_class, type) or not issubclass(encoder_class, ContentEncoder):
        raise TypeError(f"{dotted_path!r} is not a ContentEncoder class.")

    return encoder_class


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
