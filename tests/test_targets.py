import math
from types import SimpleNamespace

import torch

from ssl_probe.dataset import FrameDataset
from ssl_probe.targets.target import HNR, SPECTRAL_FLUX, TARGETS
from ssl_probe.targets.task import RegressionTask, TargetStandardization

EGEMAPS_LLD_SOURCES = {
    "Loudness_sma3",
    "alphaRatio_sma3",
    "hammarbergIndex_sma3",
    "slope0-500_sma3",
    "slope500-1500_sma3",
    "spectralFlux_sma3",
    "mfcc1_sma3",
    "mfcc2_sma3",
    "mfcc3_sma3",
    "mfcc4_sma3",
    "F0semitoneFrom27.5Hz_sma3nz",
    "jitterLocal_sma3nz",
    "shimmerLocaldB_sma3nz",
    "HNRdBACF_sma3nz",
    "logRelF0-H1-H2_sma3nz",
    "logRelF0-H1-A3_sma3nz",
    "F1frequency_sma3nz",
    "F1bandwidth_sma3nz",
    "F1amplitudeLogRelF0_sma3nz",
    "F2frequency_sma3nz",
    "F2bandwidth_sma3nz",
    "F2amplitudeLogRelF0_sma3nz",
    "F3frequency_sma3nz",
    "F3bandwidth_sma3nz",
    "F3amplitudeLogRelF0_sma3nz",
}


def test_target_registry_covers_all_egemaps_lld_sources() -> None:
    assert {target.source for target in TARGETS.values()} == EGEMAPS_LLD_SOURCES
    assert "f2" in TARGETS
    assert "f3" in TARGETS
    assert "semitone" in TARGETS
    assert "f1_bin" in TARGETS


def test_spectral_flux_masks_invalid_values_before_log_transform() -> None:
    values, valid = SPECTRAL_FLUX.apply(torch.tensor([-1.0, 0.0, 1.0, float("nan")]))

    assert torch.equal(valid, torch.tensor([False, True, True, False]))
    assert torch.allclose(values, torch.tensor([0.0, 0.0, math.log(2.0), 0.0]))


def test_hnr_retains_negative_db_values_but_rejects_zero_fill() -> None:
    _, valid = HNR.apply(torch.tensor([-4.0, 0.0, 8.0, float("inf")]))

    assert torch.equal(valid, torch.tensor([True, False, True, False]))


def test_standardized_regression_uses_z_score_loss_and_raw_scale_metrics() -> None:
    standardization = TargetStandardization(mean=10.0, std=2.0, count=100)
    task = RegressionTask(standardization=standardization)

    result = task.compute(
        output=torch.tensor([[1.0], [-1.0]]),
        target=torch.tensor([12.0, 8.0]),
    )

    assert result.loss.item() == 0.0
    assert torch.equal(result.prediction, torch.tensor([12.0, 8.0]))
    assert torch.equal(result.metric_input, torch.tensor([12.0, 8.0]))
    assert torch.equal(result.metric_target, torch.tensor([12.0, 8.0]))


def test_training_standardization_uses_only_valid_transformed_values() -> None:
    dataset = FrameDataset.__new__(FrameDataset)
    dataset.target = TARGETS["spectral_flux"]
    dataset.base_dataset = SimpleNamespace(
        rows=[
            SimpleNamespace(utt_id="first"),
            SimpleNamespace(utt_id="second"),
        ]
    )
    dataset.smile_features = {
        "first": torch.tensor([0.0, 1.0, -1.0]),
        "second": torch.tensor([3.0, float("nan")]),
    }
    dataset.stores = {}

    standardization = dataset.compute_target_standardization()
    expected = torch.log1p(torch.tensor([0.0, 1.0, 3.0]))

    assert standardization.count == 3
    assert standardization.mean == expected.mean().item()
    assert standardization.std == expected.double().std(correction=0).item()
