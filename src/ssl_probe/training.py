from __future__ import annotations

import json
from dataclasses import dataclass, replace
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
        self.save_hyperparameters(
            ignore=["probe", "target", "content_encoder", "train_dataset", "val_dataset"]
        )

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
        target = batch.resources[self.target.name].values[:, 0].to(x.device)
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


class OnlineProbeTrainingModule(ProbeTrainingModule):
    """Run the content encoder once per audio batch inside the training step."""

    def __init__(
        self,
        *,
        content_encoder,
        train_dataset,
        val_dataset=None,
        train_encoder=False,
        encoder_lr=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if encoder_lr <= 0:
            raise ValueError("encoder_lr must be positive.")
        self.content_encoder = content_encoder
        self.content_encoder.requires_grad_(train_encoder)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_encoder = train_encoder
        self.encoder_lr = encoder_lr
        # Frozen encoder weights can be reconstructed from the recorded config.
        if not train_encoder:
            self.checkpoint_exclude_prefixes = ("content_encoder.",)

    def train(self, mode=True):
        super().train(mode)
        if not self.train_encoder:
            self.content_encoder.eval()
        return self

    def configure_optimizers(self):
        encoder_ids = {id(p) for p in self.content_encoder.parameters()}
        factory = self.optimization.optimizer

        def grouped_optimizer(parameters, **kwargs):
            parameters = list(parameters)
            groups = [
                {"params": [p for p in parameters if id(p) not in encoder_ids], "name": "probe"}
            ]
            encoder_parameters = [p for p in parameters if id(p) in encoder_ids]
            if encoder_parameters:
                groups.append(
                    {"params": encoder_parameters, "lr": self.encoder_lr, "name": "encoder"}
                )
            return factory(groups, **kwargs)

        return replace(self.optimization, optimizer=grouped_optimizer).configure(
            self.parameters(),
            total_steps=self.trainer.estimated_stepping_batches,
        )

    def training_step(self, batch, batch_idx):
        output = self._shared_step(batch, "train")
        return output.loss if output.batch_size else None

    @property
    def grad_norm_modules(self):
        return {"encoder": self.content_encoder, "probe": self.probe}

    def _shared_step(self, batch, stage):
        dataset = self.train_dataset if stage == "train" else self.val_dataset
        # ContentEncoder adapters keep an explicit device attribute for processor inputs.
        self.content_encoder.device = self.device
        with torch.set_grad_enabled(
            torch.is_grad_enabled() and stage == "train" and self.train_encoder
        ):
            content = self.content_encoder(batch)
            if (
                stage == "train"
                and self.train_encoder
                and torch.is_grad_enabled()
                and not content.values.requires_grad
            ):
                raise RuntimeError("The selected encoder backend does not preserve autograd.")
        frames = dataset.frames_from_content(list(batch), content)
        if not frames:
            # Keep a differentiable zero for automatic optimization on unlabelled batches.
            loss = sum(p.sum() * 0 for p in self.probe.parameters())
            return ProbeOutput(loss=loss, prediction=content.values.new_empty(0), batch_size=0)
        return super()._shared_step(AudioBatch.from_samples(frames), stage)
