from __future__ import annotations

import json
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import torch
from quick_convert.components.layers import LayerWeightedSum
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

    def __init__(
        self,
        probe: Probe,
        optimization: Optimization,
        target: FrameTarget,
        layer_log_interval: int = 1000,
    ) -> None:
        super().__init__(optimization)
        if layer_log_interval <= 0:
            raise ValueError("layer_log_interval must be positive.")

        self.probe = probe
        self.target = target
        self.layer_log_interval = layer_log_interval
        self.train_metrics = target.task.make_metrics().clone(prefix="train/")
        self.val_metrics = target.task.make_metrics().clone(prefix="val/")
        self.save_hyperparameters(ignore=["probe", "target"])

    @property
    def layer_fusion(self) -> LayerWeightedSum | None:
        transform = self.probe.feature_transform
        return transform if isinstance(transform, LayerWeightedSum) else None

    def normalized_layer_weights(self) -> torch.Tensor | None:
        if self.layer_fusion is None:
            return None
        return self.layer_fusion.weights.softmax(dim=-1).squeeze(0)

    def _log_layer_weights(self) -> None:
        weights = self.normalized_layer_weights()
        if weights is None:
            return

        self.log_dict(
            {f"layer_weights/layer_{index:02d}": weight for index, weight in enumerate(weights)},
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )
        self.media_logger.log_bar(
            "layer_weights",
            weights,
            "layer",
            "weight",
            item_labels=list(range(len(weights))),
            step=self.global_step,
        )

    def export_layer_weights(self, path: PathLike) -> Path | None:
        weights = self.normalized_layer_weights()
        if weights is None or self.layer_fusion is None:
            return None

        logits = self.layer_fusion.weights.squeeze(0)
        weights_list = weights.detach().cpu().tolist()
        payload = {
            "type": "weighted-sum",
            "num_layers": len(weights_list),
            "logits": logits.detach().cpu().tolist(),
            "normalized_weights": weights_list,
            "layers": [
                {"layer": index, "weight": weight} for index, weight in enumerate(weights_list)
            ],
        }

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n")
        return output_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.probe(x)

    def _shared_step(self, batch: AudioBatch, stage: str) -> ProbeOutput:
        x = batch.resources["content"].values
        target = batch.resources[self.target.name].values[:, 0]
        result = self.target.task.compute(self(x), target)

        self.log(
            f"{stage}/loss",
            result.loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch),
        )

        metrics = self.train_metrics if stage == "train" else self.val_metrics
        metrics.update(result.metric_input, result.metric_target)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        if (
            stage == "train"
            and self.layer_fusion is not None
            and self.global_step % self.layer_log_interval == 0
        ):
            self._log_layer_weights()

        return ProbeOutput(
            loss=result.loss,
            prediction=result.prediction,
            batch_size=len(batch),
        )
