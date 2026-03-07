from torch import nn
import torch


class ValueNetwork2(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = int(input_dim)
        self.feature_dim = self.input_dim * 3
        hidden1 = max(256, self.feature_dim)
        hidden2 = max(192, self.feature_dim // 2)
        hidden3 = max(128, self.feature_dim // 3)

        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.SiLU(),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.SiLU(),
            nn.Linear(hidden2, hidden3),
            nn.LayerNorm(hidden3),
            nn.SiLU(),
            nn.Linear(hidden3, 1),
        )

    def _features(self, x):
        x = x.float()
        x_sq = x * x
        x_abs = torch.abs(x)
        return torch.cat([x, x_sq, x_abs], dim=-1)

    def forward(self, x):
        features = self._features(x)
        return self.net(features)
