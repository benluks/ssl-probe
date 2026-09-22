import json
import math
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from quick_convert.components.layers import LayerWeightedSum

from ssl_probe.probe import Probe
from ssl_probe.training import ProbeTrainingModule


class DummyMetrics:
    def clone(self, prefix: str):
        return self


class DummyTask:
    def make_metrics(self):
        return DummyMetrics()


class DummyTarget:
    task = DummyTask()


class RecordingMediaLogger:
    def __init__(self) -> None:
        self.bar = None

    def log_bar(
        self,
        key,
        values,
        item_name,
        value_name,
        *,
        item_labels,
        step,
    ) -> None:
        self.bar = {
            "key": key,
            "values": values.detach().clone(),
            "item_name": item_name,
            "value_name": value_name,
            "item_labels": item_labels,
            "step": step,
        }


def make_module(with_fusion: bool = True) -> ProbeTrainingModule:
    fusion = LayerWeightedSum(num_layers=2) if with_fusion else None
    probe = Probe(
        input_dim=4,
        output_dim=1,
        feature_transform=fusion,
    )
    return ProbeTrainingModule(
        probe=probe,
        optimization=None,
        target=DummyTarget(),
    )


def test_normalized_layer_weights_and_json_export(tmp_path) -> None:
    module = make_module()
    with torch.no_grad():
        module.layer_fusion.weights.copy_(torch.tensor([[0.0, math.log(3.0)]]))

    weights = module.normalized_layer_weights()
    output_path = module.export_layer_weights(tmp_path / "layer_weights.json")
    payload = json.loads(output_path.read_text())

    assert torch.allclose(weights, torch.tensor([0.25, 0.75]))
    assert payload["type"] == "weighted-sum"
    assert payload["num_layers"] == 2
    assert payload["normalized_weights"] == [0.25, 0.75]
    assert payload["layers"] == [
        {"layer": 0, "weight": 0.25},
        {"layer": 1, "weight": 0.75},
    ]


def test_layer_weight_logging_uses_quick_convert_media_logger() -> None:
    module = make_module()
    media_logger = RecordingMediaLogger()
    module._media_logger = media_logger
    module.log_dict = Mock()
    module._trainer = SimpleNamespace(global_step=7)

    module._log_layer_weights()

    module.log_dict.assert_called_once()
    assert media_logger.bar["key"] == "layer_weights"
    assert media_logger.bar["item_labels"] == [0, 1]
    assert media_logger.bar["step"] == 7
    assert torch.allclose(
        media_logger.bar["values"],
        torch.tensor([0.5, 0.5]),
    )


def test_single_layer_probe_has_no_weight_export(tmp_path) -> None:
    module = make_module(with_fusion=False)
    output_path = tmp_path / "layer_weights.json"

    assert module.normalized_layer_weights() is None
    assert module.export_layer_weights(output_path) is None
    assert not output_path.exists()
