import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from src.utils.logger import logger
import os


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(ActorCritic, self).__init__()
        self.affine = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        
        self.action_layer = nn.Linear(128, action_dim)
        self.value_layer = nn.Linear(128, 1)
        
        self.logprobs = []
        self.state_values = []
        self.rewards = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        state = F.relu(self.affine(state))
        
        state_value = self.value_layer(state)
        
        action_probs = F.softmax(self.action_layer(state))
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        
        return action.item()
    
    def calculateLoss(self, gamma=0.99):
        if not (self.logprobs and self.state_values and self.rewards):
            logger.error("Warning: Empty memory buffers!")
            return torch.tensor(0.0, device=self.device)
        

        # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)
                
        # normalizing the rewards:
       
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward  - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward.unsqueeze(0))
            loss += (action_loss + value_loss)   
        return loss
    
    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]


    def save(self, path="src\\models\\actor_critic", file="actor_critic.pth"):
        """
        Save the model parameters to the specified path.
        """
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, file))
        logger.info(f"Model saved to {path}")

    def load(self, path="src\\models\\actor_critic", file="actor_critic.pth", map_location=None):
        """
        Load the model parameters from the specified path.
        """
        path = os.path.join(path, file)
        if not os.path.exists(path):
            logger.error(f"Model file {path} does not exist.")
            return
        device = map_location if map_location is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(torch.load(path, map_location=device))
        logger.info(f"Model loaded from {path}")


