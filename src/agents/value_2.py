from torch import nn
import torch


class ValueNetwork2(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = int(input_dim)
        self.feature_dim = self.input_dim * 3
        self.out = nn.Linear(self.feature_dim, 1)

    def _features(self, x):
        x = x.float()
        x_sq = x * x
        x_abs = torch.abs(x)
        return torch.cat([x, x_sq, x_abs], dim=-1)

    def forward(self, x):
        features = self._features(x)
        return self.out(features)
