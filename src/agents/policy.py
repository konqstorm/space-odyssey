import torch.nn as nn
import torch


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_thrust = nn.Linear(hidden_dim, hidden_dim)
        self.head_thrust = nn.Linear(hidden_dim, 2)
        self.fc_torque = nn.Linear(hidden_dim, hidden_dim)
        self.head_torque = nn.Linear(hidden_dim, 2)
        self._init_policy_biases()

    def _init_policy_biases(self):
        # Более нейтральный старт: не "душим" тягу и оставляем умеренный шум.
        nn.init.constant_(self.head_thrust.bias[0], 0.0)
        nn.init.constant_(self.head_torque.bias[0], 0.0)
        nn.init.constant_(self.head_thrust.bias[1], -0.5)
        nn.init.constant_(self.head_torque.bias[1], -0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        x_thrust = torch.relu(self.fc_thrust(x))
        x_torque = torch.relu(self.fc_torque(x))

        out_thrust = self.head_thrust(x_thrust)
        out_torque = self.head_torque(x_torque)
        
        # Mean в "raw"-пространстве; ограничение [-1, 1] делаем через tanh уже при сэмплинге action.
        mean = torch.cat([
            out_thrust[:, 0:1],
            out_torque[:, 0:1]
        ], dim=-1)
        log_std = torch.cat([
            out_thrust[:, 1:2],
            out_torque[:, 1:2]
        ], dim=-1)
        log_std = torch.clamp(log_std, -4, 1)
        return mean, log_std
