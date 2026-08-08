# pitch_probe/module.py

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from quick_convert.data import AudioBatch
from quick_convert.pipelines.training import BaseTrainingModule, Optimization
from torch import nn
from torchmetrics.regression import R2Score


@dataclass
class ProbeOutput:
    loss: torch.Tensor
    prediction: torch.Tensor
    batch_size: int


class Probe(BaseTrainingModule):
    def __init__(
        self,
        input_dim: int,
        optimization: Optimization,
    ) -> None:
        super().__init__(optimization)

        self.probe = nn.Linear(input_dim, 1)
        self.r2 = R2Score()

    def _shared_step(
        self,
        batch: AudioBatch,
        stage: str,
    ) -> ProbeOutput:
        ssl = batch.resources["content"].values
        pitch = batch.resources["pitch"].values.squeeze(-1)

        pred = self.probe(ssl).squeeze(-1)

        loss = F.mse_loss(pred, pitch)

        self.log(
            f"{stage}/loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{stage}/r2",
            self.r2(pred.detach(), pitch.detach()),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch),
        )

        return ProbeOutput(loss=loss, prediction=pred, batch_size=len(batch))
