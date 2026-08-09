from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from quick_convert.utils import ConfigurableDevice
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection, R2Score
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    MulticlassAccuracy,
    MulticlassF1Score,
)


@dataclass
class TaskOutput:
    loss: torch.Tensor
    prediction: torch.Tensor
    metric_input: torch.Tensor
    metric_target: torch.Tensor


class ProbeTask(ABC):
    output_dim: int
    device: ConfigurableDevice

    @abstractmethod
    def compute(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> TaskOutput: ...

    @abstractmethod
    def make_metrics(self) -> MetricCollection: ...


class RegressionTask(ProbeTask):
    output_dim = 1

    def compute(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> TaskOutput:
        prediction = output.squeeze(-1)
        target = target.squeeze(-1)

        return TaskOutput(
            loss=F.mse_loss(prediction, target),
            prediction=prediction,
            metric_input=prediction,
            metric_target=target,
        )

    def make_metrics(self) -> MetricCollection:
        return MetricCollection(
            {
                "mae": MeanAbsoluteError(),
                "mse": MeanSquaredError(),
                "rmse": MeanSquaredError(squared=False),
                "r2": R2Score(),
            }
        )


class BinaryClassificationTask(ProbeTask):
    output_dim = 1

    def compute(self, output, target):
        # logits have 1 extra dim
        logits = output.squeeze(-1)
        target = target.float()

        self.device = target.device

        probability = logits.sigmoid()

        return TaskOutput(
            loss=F.binary_cross_entropy_with_logits(logits, target.float()),
            prediction=(logits >= 0).long(),
            metric_input=probability,
            metric_target=target,
        )

    def make_metrics(self):
        return MetricCollection(
            {
                "accuracy": BinaryAccuracy(),
                "auroc": BinaryAUROC(),
                "f1": BinaryF1Score(),
            }
        )


class ClassificationTask(ProbeTask):
    def __init__(
        self,
        num_classes: int,
    ) -> None:
        self.num_classes = num_classes
        self.output_dim = num_classes

    def compute(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> TaskOutput:
        target = target.squeeze(-1).long()

        probabilities = output.softmax(dim=-1)

        return TaskOutput(
            loss=F.cross_entropy(
                output,
                target,
            ),
            prediction=output.argmax(dim=-1),
            metric_input=probabilities,
            metric_target=target,
        )

    def make_metrics(self) -> MetricCollection:
        return MetricCollection(
            {
                "accuracy": MulticlassAccuracy(
                    num_classes=self.num_classes,
                ),
                "f1": MulticlassF1Score(
                    num_classes=self.num_classes,
                    average="macro",
                ),
            }
        )
