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


MIN_SEMITONE = 12
MAX_SEMITONE = 60

F1_MIN_HZ = 150
F1_MAX_HZ = 1500
FREQ_BINS_PER_OCTAVE = 24

F1_NUM_CLASSES = math.ceil(FREQ_BINS_PER_OCTAVE * math.log2(F1_MAX_HZ / F1_MIN_HZ))


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
            transformed = values.clone()
            transformed[mask] = self.transform(
                values[mask],
                **transform_kwargs,
            )
            values = transformed

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
    return x != 0


def to_voiced(x: torch.Tensor) -> torch.Tensor:
    return (x > 0)


def to_semitone_class(x: torch.Tensor) -> torch.Tensor:
    return x.round().long() - MIN_SEMITONE


def valid_pitch_class(x: torch.Tensor) -> torch.Tensor:
    return (x >= MIN_SEMITONE) & (x <= MAX_SEMITONE)


def positive(x: torch.Tensor) -> torch.Tensor:
    return x > 0


def finite(x: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(x)


def log_positive(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x)


def log1p(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(x)


def frequency_range(min_hz: float, max_hz: float):
    def mask(x: torch.Tensor) -> torch.Tensor:
        return (x >= min_hz) & (x <= max_hz)

    return mask


def frequency_bins(min_hz: float, bins_per_octave: int):
    def transform(x: torch.Tensor) -> torch.Tensor:
        return torch.floor(bins_per_octave * torch.log2(x / min_hz)).long()

    return transform


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


LOG_F0 = FrameTarget(
    name="logf0",
    source="F0semitoneFrom27.5Hz_sma3nz",
    task=RegressionTask(),
    transform=semitone_to_log_hz,
    valid_mask=valid_pitch_class,
)


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
    transform=log_positive,
    valid_mask=frequency_range(F1_MIN_HZ, F1_MAX_HZ),
)

F2 = FrameTarget(
    name="f2",
    source="F2frequency_sma3nz",
    task=RegressionTask(),
    transform=log_positive,
    valid_mask=positive,
)

F3 = FrameTarget(
    name="f3",
    source="F3frequency_sma3nz",
    task=RegressionTask(),
    transform=log_positive,
    valid_mask=positive,
)

F1_BIN = FrameTarget(
    name="f1_bin",
    source="F1frequency_sma3nz",
    task=BinClassificationTask(F1_NUM_CLASSES),
    transform=frequency_bins(F1_MIN_HZ, FREQ_BINS_PER_OCTAVE),
    valid_mask=frequency_range(F1_MIN_HZ, F1_MAX_HZ),
)


VOICED = FrameTarget(
    name="voiced",
    source="F0semitoneFrom27.5Hz_sma3nz",
    task=BinaryClassificationTask(),
    transform=to_voiced,
)


JITTER = FrameTarget(
    name="jitter",
    source="jitterLocal_sma3nz",
    task=RegressionTask(),
    transform=log1p,
    valid_mask=positive,
)

SHIMMER = FrameTarget(
    name="shimmer",
    source="shimmerLocaldB_sma3nz",
    task=RegressionTask(),
    valid_mask=nonzero,
)

SPECTRAL_FLUX = FrameTarget(
    name="spectral_flux",
    source="spectralFlux_sma3",
    task=RegressionTask(),
    transform=log1p,
)

HNR = FrameTarget(
    name="hnr",
    source="HNRdBACF_sma3nz",
    task=RegressionTask(),
    valid_mask=nonzero,
)

ALPHA_RATIO = FrameTarget(
    name="alpha_ratio",
    source="alphaRatio_sma3",
    task=RegressionTask(),
)

HAMMARBERG = FrameTarget(
    name="hammarberg",
    source="hammarbergIndex_sma3",
    task=RegressionTask(),
)

TARGETS = {
    "logf0": LOG_F0,
    "smn_logf0": SPEAKER_NORMALIZED_LOG_F0,
    "voiced": VOICED,
    "loudness": LOUDNESS,
    "f1": F1,
    "f1_bin": F1_BIN,
    "semitone": SEMITONE,
    "jitter": JITTER,
    "shimmer": SHIMMER,
    "spectral_flux": SPECTRAL_FLUX,
    "hnr": HNR,
    "alpha_ratio": ALPHA_RATIO,
    "hammarberg": HAMMARBERG,
}
