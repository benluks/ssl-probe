from __future__ import annotations

from dataclasses import dataclass

import torch
from quick_convert.data import AudioBatch
from quick_convert.training.lightning.modules.base import BaseTrainingModule
from quick_convert.training.lightning.optim import Optimization
from torch import nn

from .targets import FrameTarget

NONLINEARITIES = {"relu": nn.ReLU, "gelu": nn.GELU, "none": nn.Identity}


class Probe(nn.Module):
    """A lightweight probe that maps frame-level representations to predictions."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: list[int] | None = None,
        nonlinearity: str = "gelu",
    ) -> None:
        super().__init__()

        hidden_dim = hidden_dim or []
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]

        layers = []
        in_dim = input_dim

        output_dims = [*hidden_dim, output_dim]
        for i, dim in enumerate(output_dims):
            layers.append(nn.Linear(in_dim, dim))
            if i != len(output_dims) - 1:
                layers.append(NONLINEARITIES[nonlinearity]())
            in_dim = dim

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass
class ProbeOutput:
    loss: torch.Tensor
    prediction: torch.Tensor
    batch_size: int


class ProbeTrainingModule(BaseTrainingModule):
    """Lightning training adapter for a probe and its target task."""

    def __init__(
        self,
        probe: Probe,
        optimization: Optimization,
        target: FrameTarget,
    ) -> None:
        super().__init__(optimization)

        self.probe = probe
        self.target = target

        self.train_metrics = target.task.make_metrics().clone(prefix="train/")
        self.val_metrics = target.task.make_metrics().clone(prefix="val/")

        self.save_hyperparameters(ignore=["probe", "target"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.probe(x)

    def _shared_step(
        self,
        batch: AudioBatch,
        stage: str,
    ) -> ProbeOutput:
        x = batch.resources["content"].values
        # Flatten for context sizes larger than 1.
        x = x.flatten(1)
        target = batch.resources[self.target.name].values[:, 0]

        output = self.probe(x)
        result = self.target.task.compute(output, target)

        self.log(
            f"{stage}/loss",
            result.loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch),
        )

        metrics = self.train_metrics if stage == "train" else self.val_metrics

        metrics.update(
            result.metric_input,
            result.metric_target,
        )

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return ProbeOutput(
            loss=result.loss,
            prediction=result.prediction,
            batch_size=len(batch),
        )
