from __future__ import annotations

import torch
from torch import nn

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
        for i, dim in enumerate([*hidden_dim, output_dim]):
            layers.append(nn.Linear(in_dim, dim))
            if i != len(hidden_dim):
                layers.append(NONLINEARITIES[nonlinearity]())
            in_dim = dim

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
