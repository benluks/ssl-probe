from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from quick_convert.data import AudioBatch, AudioSample
from quick_convert.training.lightning.optim import Optimization
from torch import nn

from ssl_probe.dataset import FrameDataset
from ssl_probe.probe import Probe
from ssl_probe.targets import FrameTarget, RegressionTask
from ssl_probe.training import OnlineProbeTrainingModule


class TinyEncoder(nn.Module):
    feature_dim = 2

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        values = self.scale * torch.ones(len(batch), 5, 2)
        return SimpleNamespace(values=values, lengths=torch.full((len(batch),), 5), frame_hz=50.0)


def make_dataset(encoder, *, valid=True):
    dataset = FrameDataset.__new__(FrameDataset)
    dataset.content_encoder = encoder
    dataset.content_frame_hz = 50.0
    dataset.preserve_layers = False
    dataset.context_size = 3
    dataset.context_radius = 1
    dataset.target = FrameTarget(
        name="test",
        source="test",
        task=RegressionTask(),
        valid_mask=lambda x: x > 0,
    )
    dataset.smile_features = SimpleStore(valid=valid)
    dataset.stores = {}
    return dataset


class SimpleStore:
    frame_hz = 100.0

    def __init__(self, valid):
        self.valid = valid

    def __getitem__(self, key):
        return torch.arange(10).float() + 1 if self.valid else torch.zeros(10)


def make_batch():
    return AudioBatch.from_samples(
        [
            AudioSample(
                utt_id="a", path=Path("a.wav"), waveform=torch.ones(1, 1600), sample_rate=16000
            ),
        ]
    )


def make_module(train_encoder, *, valid=True):
    encoder = TinyEncoder()
    dataset = make_dataset(encoder, valid=valid)
    module = OnlineProbeTrainingModule(
        probe=Probe(input_dim=6, output_dim=1),
        target=dataset.target,
        optimization=Optimization(optimizer_kwargs={"lr": 0.01}, lr_scheduler=None),
        content_encoder=encoder,
        train_dataset=dataset,
        val_dataset=dataset,
        train_encoder=train_encoder,
        encoder_lr=0.001,
    )
    module.log = Mock()
    module.log_dict = Mock()
    module._trainer = SimpleNamespace(estimated_stepping_batches=2, global_step=0)
    return module


@pytest.mark.parametrize("train_encoder", [False, True])
def test_two_optimizer_steps_and_checkpoint_contract(train_encoder):
    module = make_module(train_encoder)
    module.train()
    assert module.content_encoder.training == train_encoder
    optimizer = module.configure_optimizers()
    assert optimizer.param_groups[0]["lr"] == 0.01
    if train_encoder:
        assert optimizer.param_groups[1]["lr"] == 0.001
    before = module.content_encoder.scale.detach().clone()
    for _ in range(2):
        optimizer.zero_grad()
        output = module._shared_step(make_batch(), "train")
        assert output.batch_size == 3
        output.loss.backward()
        assert (module.content_encoder.scale.grad is not None) == train_encoder
        optimizer.step()
    assert (not torch.equal(before, module.content_encoder.scale)) == train_encoder
    checkpoint = {"state_dict": module.state_dict()}
    module.on_save_checkpoint(checkpoint)
    assert ("content_encoder.scale" in checkpoint["state_dict"]) == train_encoder
    restored = make_module(train_encoder)
    restored.on_load_checkpoint(checkpoint)
    restored.load_state_dict(checkpoint["state_dict"])
    torch.testing.assert_close(restored.content_encoder.scale, module.content_encoder.scale)
    with torch.no_grad():
        assert not module._shared_step(make_batch(), "val").loss.requires_grad


def test_alignment_and_context_match_existing_frame_path():
    module = make_module(True)
    batch = make_batch()
    frames = module.train_dataset.frames_from_content(list(batch), module.content_encoder(batch))
    collated = AudioBatch.from_samples(frames)
    assert collated.resources["content"].values.shape == (3, 3, 2)
    torch.testing.assert_close(
        collated.resources["test"].values[:, 0, 0], torch.tensor([3.0, 5.0, 7.0])
    )
    assert collated.resources["content"].values.requires_grad


def test_empty_target_batch_has_finite_differentiable_zero():
    module = make_module(True, valid=False)
    output = module._shared_step(make_batch(), "train")
    assert output.batch_size == 0
    assert output.loss.item() == 0
    output.loss.backward()


def test_empty_target_batch_skips_optimizer_step():
    module = make_module(True, valid=False)
    assert module.training_step(make_batch(), 0) is None


def test_cli_requires_utterance_mode_for_finetuning():
    from ssl_probe.commands.train import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--root", "/audio", "--train-encoder"])
    args = parse_args(["--root", "/audio", "--batch-mode", "utterances", "--train-encoder"])
    assert args.train_encoder


def test_lightning_fit_and_validation(tmp_path):
    import lightning as L
    from torch.utils.data import DataLoader

    module = make_module(True)
    # Use actual Lightning logging and batch transfer, rather than the unit-test mocks.
    del module.log
    del module.log_dict
    module._trainer = None
    trainer = L.Trainer(
        accelerator="cpu",
        max_steps=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        val_check_interval=1,
        default_root_dir=tmp_path,
    )
    loader = DataLoader([make_batch(), make_batch()], batch_size=None)
    before = module.content_encoder.scale.detach().clone()
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    assert trainer.global_step == 2
    assert not torch.equal(before, module.content_encoder.scale)
    assert "val/loss" in trainer.callback_metrics


def test_online_frame_sampling_preserves_gradient_and_validation_coverage():
    module = make_module(True)
    module.online_frame_sample_size = 2
    train_output = module._shared_step(make_batch(), "train")
    assert train_output.batch_size == 2
    assert module.cumulative_valid_frames == 3
    assert module.cumulative_sampled_frames == 2
    train_output.loss.backward()
    assert module.content_encoder.scale.grad is not None
    val_output = module._shared_step(make_batch(), "val")
    assert val_output.batch_size == 3


def test_online_frame_sampling_cli_validation():
    from ssl_probe.commands.train import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--root", "/audio", "--online-frame-sample-size", "2"])
    with pytest.raises(SystemExit):
        parse_args(
            ["--root", "/audio", "--batch-mode", "utterances", "--online-frame-sample-size", "0"]
        )
    args = parse_args(
        ["--root", "/audio", "--batch-mode", "utterances", "--online-frame-sample-size", "128"]
    )
    assert args.online_frame_sample_size == 128
