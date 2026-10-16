"""The Deep MLP architecture (training only; serving uses the ONNX export)."""

import numpy as np
import torch
import torch.nn as nn

HIDDEN_DIMS = [128, 64, 32]
DROPOUT = 0.5


class DeepMLP(nn.Module):
    """Fully-connected deep neural network for binary commit classification."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RiskScorer(nn.Module):
    """Scaling, the MLP and softmax as one graph: engineered features in,
    probability of the buggy class out. This is what gets exported."""

    def __init__(self, mlp: DeepMLP, mean: np.ndarray, scale: np.ndarray):
        super().__init__()
        self.mlp = mlp
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.mlp((features - self.mean) / self.scale)
        return torch.softmax(logits, dim=1)[:, 1]
