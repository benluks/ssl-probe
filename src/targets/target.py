from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from .task import (
    BinaryClassificationTask,
    BinClassificationTask,
    ProbeTask,
    RegressionTask,
)

TensorTransform = Callable[[torch.Tensor], torch.Tensor]
TensorMask = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class FrameTarget:
    name: str
    source: str
    task: ProbeTask
    transform: TensorTransform | None = None
    valid_mask: TensorMask | None = None
    resources: tuple[str, ...] = ()
    transform_kwargs: Callable | None = None

    def apply(
        self,
        values: torch.Tensor,
        **transform_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform raw feature values and return values + valid mask."""

        mask = torch.ones(
            len(values),
            dtype=torch.bool,
            device=values.device,
        )

        if self.valid_mask is not None:
            mask &= self.valid_mask(values)

        if self.transform is not None:
            values = self.transform(values, **transform_kwargs)

        return values, mask


def semitone_to_log_hz(x: torch.Tensor) -> torch.Tensor:
    return math.log(27.5) + x * (math.log(2.0) / 12.0)


def semitone_to_speaker_normalized_log_hz(
    x: torch.Tensor,
    *,
    speaker_mean_log_f0: float | torch.Tensor,
    **_,
) -> torch.Tensor:
    log_f0 = semitone_to_log_hz(x)
    return log_f0 - torch.as_tensor(
        speaker_mean_log_f0,
        device=x.device,
        dtype=log_f0.dtype,
    )


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


def nonzero(x: torch.Tensor) -> torch.Tensor:
    return x > 0


def to_voiced(x: torch.Tensor) -> torch.Tensor:
    return (x > 0).long()


MIN_SEMITONE = 12
MAX_SEMITONE = 60


def to_semitone_class(x: torch.Tensor) -> torch.Tensor:
    return x.round().long() - MIN_SEMITONE


def valid_pitch_class(x: torch.Tensor) -> torch.Tensor:
    return (x >= MIN_SEMITONE) & (x <= MAX_SEMITONE)


LOG_F0 = FrameTarget(
    name="logf0",
    source="F0semitoneFrom27.5Hz_sma3nz",
    task=RegressionTask(),
    transform=semitone_to_log_hz,
    valid_mask=valid_pitch_class,
)


def speaker_log_f0_kwargs(sample, stores):
    if "speaker_stats" not in stores:
        raise ValueError("speaker_normalized_logf0 requires speaker statistics.")

    spk_id = sample.resources["spk_id"].value

    try:
        speaker_mean = stores["speaker_stats"][spk_id]
    except KeyError:
        raise KeyError(f"No speaker statistics found for speaker {spk_id!r}.")

    return {
        "speaker_mean_log_f0": speaker_mean,
    }


SPEAKER_NORMALIZED_LOG_F0 = FrameTarget(
    name="speaker_normalized_logf0",
    source="F0semitoneFrom27.5Hz_sma3nz",
    task=RegressionTask(),
    transform=semitone_to_speaker_normalized_log_hz,
    valid_mask=valid_pitch_class,
    resources=("spk_id",),
    transform_kwargs=speaker_log_f0_kwargs,
)


SEMITONE = FrameTarget(
    name="semitone",
    source="F0semitoneFrom27.5Hz_sma3nz",
    task=BinClassificationTask(MAX_SEMITONE - MIN_SEMITONE + 1),
    transform=to_semitone_class,
    valid_mask=valid_pitch_class,
)

LOUDNESS = FrameTarget(
    name="loudness",
    source="Loudness_sma3",
    task=RegressionTask(),
)

F1 = FrameTarget(
    name="f1",
    source="F1frequency_sma3nz",
    task=RegressionTask(),
)


VOICED = FrameTarget(
    name="voiced",
    source="F0semitoneFrom27.5Hz_sma3nz",
    task=BinaryClassificationTask(),
    transform=to_voiced,
)


TARGETS = {
    "logf0": LOG_F0,
    "smn_logf0": SPEAKER_NORMALIZED_LOG_F0,
    "voiced": VOICED,
    "loudness": LOUDNESS,
    "f1": F1,
    "semitone": SEMITONE,
}
