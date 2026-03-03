from torch import nn
import torch


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        #self.fc1 = nn.Linear(input_dim, hidden_dim)
        #self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        #self.out = nn.Linear(hidden_dim, 1)
        self.out = nn.Linear(input_dim, 1)


    def forward(self, x):
        #x = torch.relu(self.fc1(x))
        #x = torch.relu(self.fc2(x))
        return self.out(x)
