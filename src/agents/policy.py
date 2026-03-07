import torch.nn as nn
import torch


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, policy_variant="shallow"):
        super().__init__()
        self.policy_variant = policy_variant
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        if self.policy_variant == "deep":
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.fc3 = None
        self.fc_thrust = nn.Linear(hidden_dim, hidden_dim)
        if self.policy_variant == "deep":
            self.fc_thrust2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc_torque2 = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.fc_thrust2 = None
            self.fc_torque2 = None
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
        if self.fc3 is not None:
            x = torch.relu(self.fc3(x))

        x_thrust = torch.relu(self.fc_thrust(x))
        if self.fc_thrust2 is not None:
            x_thrust = torch.relu(self.fc_thrust2(x_thrust))
        x_torque = torch.relu(self.fc_torque(x))
        if self.fc_torque2 is not None:
            x_torque = torch.relu(self.fc_torque2(x_torque))

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
