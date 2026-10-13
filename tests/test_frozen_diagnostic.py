from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from quick_convert.data import AudioBatch, AudioSample
from torch import nn

from ssl_probe.commands.diagnose_frozen import (
    compare_frames,
    parse_args,
    run_diagnostic,
    tensor_check,
)
from ssl_probe.dataset import FrameDataset
from ssl_probe.probe import Probe
from ssl_probe.targets import FrameTarget, RegressionTask, TargetStandardization


class Encoder(nn.Module):
    feature_dim = 2
    frame_hz = 50
    sample_rate = 16000

    def __init__(self, padding_sensitive=False):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.device = torch.device("cpu")
        self.padding_sensitive = padding_sensitive

    def forward(self, batch):
        samples = list(batch)
        lengths = torch.tensor([sample.waveform.shape[-1] // 200 for sample in samples])
        max_frames = int(lengths.max())
        values = self.scale * torch.ones(len(samples), max_frames, 2)
        if self.padding_sensitive:
            values = values + max_frames
        return SimpleNamespace(values=values, lengths=lengths, frame_hz=50.0)


class Store:
    frame_hz = 100

    def __getitem__(self, key):
        return torch.arange(20).float() + 1


def fixture(padding_sensitive=False):
    encoder = Encoder(padding_sensitive)
    dataset = FrameDataset.__new__(FrameDataset)
    dataset.content_encoder = encoder
    dataset.content_frame_hz = 50.0
    dataset.preserve_layers = False
    dataset.context_size = 3
    dataset.context_radius = 1
    dataset.target = FrameTarget(name="test", source="test", task=RegressionTask())
    dataset.smile_features = Store()
    dataset.stores = {}
    dataset.frame_lengths = [5, 7]
    dataset.base_dataset = SimpleNamespace(collate_fn=AudioBatch.from_samples)
    samples = [
        AudioSample(
            utt_id=str(i), path=Path(f"{i}.wav"), waveform=torch.ones(1, size), sample_rate=16000
        )
        for i, size in enumerate((1000, 1400))
    ]
    return dataset, samples


@pytest.mark.parametrize("standardize", [False, True])
def test_real_training_paths_match_features_gradients_and_multiple_updates(standardize):
    dataset, samples = fixture()
    if standardize:
        dataset.target = replace(
            dataset.target, task=RegressionTask(TargetStandardization(4.0, 2.0, 10))
        )
    before = dataset.content_encoder.scale.detach().clone()
    report = run_diagnostic(
        dataset, samples, audio_budget=7, reference_audio_budget=100, frame_batch_size=4, updates=3
    )
    assert report["controlled_parity_pass"]
    assert report["audio_batch_invariance_pass"]
    assert report["valid_frames"] == 8
    assert report["audio_batching_impact"]["affected_frames"] == 0
    assert report["audio_batching_impact"]["predictions"]["max_abs"] == 0
    assert len(report["controlled_updates"]) == 3
    assert torch.equal(before, dataset.content_encoder.scale)
    for update in report["controlled_updates"]:
        assert update["gradient_max_abs"] == 0
        assert update["parameter_max_abs"] == 0
        assert update["frame_lr"] == update["online_lr"]
    assert [item["frames"] for item in report["native_batching"]["frame_updates"]] == [4, 4]
    assert [item["frames"] for item in report["native_batching"]["online_updates"]] == [3, 5]


def test_diagnostic_detects_batch_dependent_encoder_without_confusing_path_parity():
    dataset, samples = fixture(padding_sensitive=True)
    report = run_diagnostic(dataset, samples, audio_budget=7, reference_audio_budget=100, updates=1)
    assert report["controlled_parity_pass"]
    assert not report["audio_batch_invariance_pass"]
    assert report["different_audio_batch_extraction"]["features"]["max_abs"] == 2
    assert report["audio_batching_impact"]["affected_frames"] == 3
    assert report["audio_batching_impact"]["features"]["max_abs"] == 2
    assert report["audio_batching_impact"]["predictions"]["max_abs"] > 0


def test_diagnostic_detects_online_feature_corruption():
    dataset, samples = fixture()
    original = dataset.frames_from_content

    def corrupted(samples, content):
        # extract_frames runs in inference mode, whereas the online path uses no_grad.
        if not torch.is_inference_mode_enabled():
            content.values = content.values + 1
        return original(samples, content)

    dataset.frames_from_content = corrupted
    report = run_diagnostic(dataset, samples, audio_budget=7, reference_audio_budget=100, updates=2)
    assert not report["controlled_parity_pass"]
    assert not report["same_audio_batch_extraction"][0]["features"]["pass"]
    assert not report["controlled_updates"][0]["gradients_pass"]


def test_checkpoint_head_is_used_for_prediction_impact(tmp_path):
    dataset, samples = fixture(padding_sensitive=True)
    probe = Probe(6, 1, [512], "gelu")
    for parameter in probe.parameters():
        parameter.data.zero_()
    path = tmp_path / "probe.ckpt"
    torch.save(
        {"state_dict": {f"probe.{key}": value for key, value in probe.state_dict().items()}}, path
    )
    report = run_diagnostic(
        dataset,
        samples,
        audio_budget=7,
        reference_audio_budget=100,
        updates=1,
        probe_checkpoint=path,
    )
    assert report["audio_batching_impact"]["features"]["max_abs"] == 2
    assert report["audio_batching_impact"]["predictions"]["max_abs"] == 0
    assert report["audio_batching_impact"]["probe_source"] == str(path)


def test_comparisons_reject_shape_and_frame_identity_mismatches():
    assert not tensor_check(torch.zeros(2), torch.zeros(3), atol=1e-5, rtol=1e-4)["pass"]
    dataset, samples = fixture()
    frames = dataset.extract_frames(samples)
    assert not compare_frames(frames, frames[:-1], dataset.target, atol=1e-5, rtol=1e-4)["pass"]


@pytest.mark.parametrize(
    "extra", [["--utterances", "0"], ["--updates", "0"], ["--context-size", "2"]]
)
def test_cli_rejects_invalid_bounds(extra):
    with pytest.raises(SystemExit):
        parse_args(["--root", "/audio", "--out-dir", "/out", *extra])
