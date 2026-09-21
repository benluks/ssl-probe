import torch

from ssl_probe.probe import Probe


def test_probe_output_shape():
    probe = Probe(input_dim=12, output_dim=3, hidden_dim=[8])
    output = probe(torch.randn(4, 12))
    assert output.shape == (4, 3)


def test_linear_probe_has_no_hidden_activation():
    probe = Probe(input_dim=12, output_dim=1)
    assert len(probe.network) == 1
