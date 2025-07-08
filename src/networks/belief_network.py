import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os



import torch
import torch.nn as nn
import torch.optim as optim

class BeliefNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.net = nn.Sequential(
            nn.Linear(config['input_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['hidden_dim']),
            nn.ReLU()
        )

        # 4 separate heads, one per ghost
        n_actions = config['output_dim']  # output_dim = n_actions per ghost
        self.heads = nn.ModuleList([nn.Linear(config['hidden_dim'], n_actions) for _ in range(4)])

        self.optimizer = optim.Adam(self.parameters(), lr=config['learning_rate'])
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = self.net(x)
        # Each head outputs [batch, n_actions]
        return [head(x) for head in self.heads]
