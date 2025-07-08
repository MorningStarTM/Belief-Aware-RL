import torch
from torch.utils.data import Dataset

class PacmanGhostActionDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data: List of episodes; each episode is a list of steps (dicts).
                  Each step should have keys 'state' and 'ghosts_last_action'.
        """
        self.states = []
        self.ghost_actions = []

        for episode in data:
            for step in episode:
                # state: flat vector, ghost_actions: list/array of 4 ints
                self.states.append(step["state"])
                self.ghost_actions.append(step["ghosts_last_action"])
        
        self.states = torch.tensor(self.states, dtype=torch.float32)
        self.ghost_actions = torch.tensor(self.ghost_actions, dtype=torch.long)
        # shape: [N, state_dim], [N, 4]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.ghost_actions[idx]





"""
from torch.utils.data import DataLoader

# Load your data as before
import pickle

with open('src\\data\\ppo_rollout_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Create dataset and loader
dataset = PacmanGhostActionDataset(data)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Test iter
for states, ghost_actions in loader:
    print(states.shape)         # (batch_size, state_dim)
    print(ghost_actions.shape)  # (batch_size, 4)
    break
"""