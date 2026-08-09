from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .task import BinaryClassificationTask, ProbeTask, RegressionTask

TensorTransform = Callable[[torch.Tensor], torch.Tensor]
TensorMask = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class FrameTarget:
    name: str
    source: str
    task: ProbeTask
    transform: callable | None = None
    valid_mask: callable | None = None

    def apply(
        self,
        values: torch.Tensor,
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
            values = self.transform(values)

        return values, mask


def semitone_to_log_hz(x: torch.Tensor) -> torch.Tensor:
    return math.log(27.5) + x * (math.log(2.0) / 12.0)


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


def nonzero(x: torch.Tensor) -> torch.Tensor:
    return x > 0


def to_voiced(x: torch.Tensor) -> torch.Tensor:
    return (x > 0).long()


LOG_F0 = FrameTarget(
    name="logf0",
    source="F0semitoneFrom27.5Hz_sma3nz",
    task=RegressionTask(),
    transform=semitone_to_log_hz,
    valid_mask=nonzero,
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


TARGETS = {"logf0": LOG_F0, "voiced": VOICED, "loudness": LOUDNESS, "f1": F1}
