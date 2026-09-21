from __future__ import annotations

from dataclasses import dataclass

import torch
from quick_convert.data import AudioBatch
from quick_convert.training.lightning.modules.base import BaseTrainingModule
from quick_convert.training.lightning.optim import Optimization

from .probe import Probe
from .targets import FrameTarget


@dataclass
class ProbeOutput:
    loss: torch.Tensor
    prediction: torch.Tensor
    batch_size: int


class ProbeTrainingModule(BaseTrainingModule):
    """Lightning training adapter for a probe and its target task."""

    def __init__(self, probe: Probe, optimization: Optimization, target: FrameTarget) -> None:
        super().__init__(optimization)
        self.probe = probe
        self.target = target
        self.train_metrics = target.task.make_metrics().clone(prefix="train/")
        self.val_metrics = target.task.make_metrics().clone(prefix="val/")
        self.save_hyperparameters(ignore=["probe", "target"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.probe(x)

    def _shared_step(self, batch: AudioBatch, stage: str) -> ProbeOutput:
        x = batch.resources["content"].values.flatten(1)
        target = batch.resources[self.target.name].values[:, 0]
        result = self.target.task.compute(self.probe(x), target)
        self.log(f"{stage}/loss", result.loss, on_step=stage == "train", on_epoch=True, prog_bar=True, batch_size=len(batch))
        metrics = self.train_metrics if stage == "train" else self.val_metrics
        metrics.update(result.metric_input, result.metric_target)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return ProbeOutput(loss=result.loss, prediction=result.prediction, batch_size=len(batch))
