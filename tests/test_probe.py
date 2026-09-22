import torch
from quick_convert.components.layers import LayerWeightedSum

from ssl_probe.probe import Probe


def test_probe_output_shape():
    probe = Probe(input_dim=12, output_dim=3, hidden_dim=[8])
    output = probe(torch.randn(4, 12))
    assert output.shape == (4, 3)


def test_probe_flattens_context_frames():
    probe = Probe(input_dim=12, output_dim=3)
    output = probe(torch.randn(4, 3, 4))
    assert output.shape == (4, 3)


def test_linear_probe_has_no_hidden_activation():
    probe = Probe(input_dim=12, output_dim=1)
    assert len(probe.network) == 1


def test_weighted_layer_fusion_receives_gradients():
    fusion = LayerWeightedSum(num_layers=2)
    probe = Probe(
        input_dim=12,
        output_dim=1,
        feature_transform=fusion,
    )
    features = torch.randn(4, 3, 2, 4)

    probe(features).sum().backward()

    assert fusion.weights.grad is not None
