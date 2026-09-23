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
            transformed_valid = self.transform(
                values[mask],
                **transform_kwargs,
            )

            transformed = torch.zeros(
                values.shape,
                dtype=transformed_valid.dtype,
                device=values.device,
            )
            transformed[mask] = transformed_valid
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


def finite(x: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(x)


def nonzero_finite(x: torch.Tensor) -> torch.Tensor:
    return finite(x) & (x != 0)


def positive_finite(x: torch.Tensor) -> torch.Tensor:
    return finite(x) & (x > 0)


def nonnegative_finite(x: torch.Tensor) -> torch.Tensor:
    return finite(x) & (x >= 0)


def to_voiced(x: torch.Tensor) -> torch.Tensor:
    return (x > 0).float()


def to_semitone_class(x: torch.Tensor) -> torch.Tensor:
    return x.round().long() - MIN_SEMITONE


def valid_pitch_class(x: torch.Tensor) -> torch.Tensor:
    return finite(x) & (x >= MIN_SEMITONE) & (x <= MAX_SEMITONE)


def log_positive(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x)


def log1p(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(x)


def frequency_range(min_hz: float, max_hz: float):
    def mask(x: torch.Tensor) -> torch.Tensor:
        return finite(x) & (x >= min_hz) & (x <= max_hz)

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
        raise KeyError(f"No speaker statistics found for speaker {spk_id!r}.") from None

    return {
        "speaker_mean_log_f0": speaker_mean,
    }


def regression_target(
    name: str,
    source: str,
    *,
    transform: TensorTransform | None = None,
    valid_mask: TensorMask = finite,
) -> FrameTarget:
    return FrameTarget(
        name=name,
        source=source,
        task=RegressionTask(),
        transform=transform,
        valid_mask=valid_mask,
    )


LOG_F0 = regression_target(
    "logf0",
    "F0semitoneFrom27.5Hz_sma3nz",
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

VOICED = FrameTarget(
    name="voiced",
    source="F0semitoneFrom27.5Hz_sma3nz",
    task=BinaryClassificationTask(),
    transform=to_voiced,
    valid_mask=finite,
)

LOUDNESS = regression_target("loudness", "Loudness_sma3")
ALPHA_RATIO = regression_target("alpha_ratio", "alphaRatio_sma3")
HAMMARBERG = regression_target("hammarberg", "hammarbergIndex_sma3")
SLOPE_0_500 = regression_target("slope_0_500", "slope0-500_sma3")
SLOPE_500_1500 = regression_target("slope_500_1500", "slope500-1500_sma3")
SPECTRAL_FLUX = regression_target(
    "spectral_flux",
    "spectralFlux_sma3",
    transform=log1p,
    valid_mask=nonnegative_finite,
)

MFCC1 = regression_target("mfcc1", "mfcc1_sma3")
MFCC2 = regression_target("mfcc2", "mfcc2_sma3")
MFCC3 = regression_target("mfcc3", "mfcc3_sma3")
MFCC4 = regression_target("mfcc4", "mfcc4_sma3")

JITTER = regression_target(
    "jitter",
    "jitterLocal_sma3nz",
    transform=log1p,
    valid_mask=positive_finite,
)
SHIMMER = regression_target(
    "shimmer",
    "shimmerLocaldB_sma3nz",
    valid_mask=positive_finite,
)
HNR = regression_target(
    "hnr",
    "HNRdBACF_sma3nz",
    valid_mask=nonzero_finite,
)
H1_H2 = regression_target(
    "h1_h2",
    "logRelF0-H1-H2_sma3nz",
    valid_mask=nonzero_finite,
)
H1_A3 = regression_target(
    "h1_a3",
    "logRelF0-H1-A3_sma3nz",
    valid_mask=nonzero_finite,
)

F1 = regression_target(
    "f1",
    "F1frequency_sma3nz",
    transform=log_positive,
    valid_mask=frequency_range(F1_MIN_HZ, F1_MAX_HZ),
)
F2 = regression_target(
    "f2",
    "F2frequency_sma3nz",
    transform=log_positive,
    valid_mask=positive_finite,
)
F3 = regression_target(
    "f3",
    "F3frequency_sma3nz",
    transform=log_positive,
    valid_mask=positive_finite,
)

F1_BIN = FrameTarget(
    name="f1_bin",
    source="F1frequency_sma3nz",
    task=BinClassificationTask(F1_NUM_CLASSES),
    transform=frequency_bins(F1_MIN_HZ, FREQ_BINS_PER_OCTAVE),
    valid_mask=frequency_range(F1_MIN_HZ, F1_MAX_HZ),
)

F1_BANDWIDTH = regression_target(
    "f1_bandwidth",
    "F1bandwidth_sma3nz",
    transform=log_positive,
    valid_mask=positive_finite,
)
F2_BANDWIDTH = regression_target(
    "f2_bandwidth",
    "F2bandwidth_sma3nz",
    transform=log_positive,
    valid_mask=positive_finite,
)
F3_BANDWIDTH = regression_target(
    "f3_bandwidth",
    "F3bandwidth_sma3nz",
    transform=log_positive,
    valid_mask=positive_finite,
)
F1_AMPLITUDE = regression_target(
    "f1_amplitude",
    "F1amplitudeLogRelF0_sma3nz",
    valid_mask=nonzero_finite,
)
F2_AMPLITUDE = regression_target(
    "f2_amplitude",
    "F2amplitudeLogRelF0_sma3nz",
    valid_mask=nonzero_finite,
)
F3_AMPLITUDE = regression_target(
    "f3_amplitude",
    "F3amplitudeLogRelF0_sma3nz",
    valid_mask=nonzero_finite,
)


TARGETS = {
    target.name: target
    for target in (
        LOG_F0,
        SPEAKER_NORMALIZED_LOG_F0,
        SEMITONE,
        VOICED,
        LOUDNESS,
        ALPHA_RATIO,
        HAMMARBERG,
        SLOPE_0_500,
        SLOPE_500_1500,
        SPECTRAL_FLUX,
        MFCC1,
        MFCC2,
        MFCC3,
        MFCC4,
        JITTER,
        SHIMMER,
        HNR,
        H1_H2,
        H1_A3,
        F1,
        F2,
        F3,
        F1_BIN,
        F1_BANDWIDTH,
        F2_BANDWIDTH,
        F3_BANDWIDTH,
        F1_AMPLITUDE,
        F2_AMPLITUDE,
        F3_AMPLITUDE,
    )
}
