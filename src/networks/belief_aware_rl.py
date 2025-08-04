import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from src.utils.logger import logger
from torch.distributions import Categorical
from datetime import datetime
from src.networks.belief_network import BeliefNet




class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.ghost_belief = []  # For ghost actions
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.ghost_belief[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def __len__(self):
        return len(self.rewards)
    

class BeliefAwareActorCritic(nn.Module):
    def __init__(self, state_dim, ghost, action_dim):
        super(BeliefAwareActorCritic, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
        self.actor = nn.Sequential(
                        nn.Linear(state_dim+ghost, 256),
                        nn.Tanh(),
                        nn.Linear(256, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, action_dim),
                        nn.Softmax(dim=-1)
                    )

        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim+ghost, 256),
                        nn.Tanh(),
                        nn.Linear(256, 128),
                        nn.Tanh(),
                        nn.Linear(128, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )



    def forward(self):
        raise NotImplementedError
    
    def act(self, state, ghost_belief):
        state = torch.cat([state, ghost_belief], dim=-1)  # [1, 456]
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        return action.detach(), action_logprob.detach(), state_val.detach()
    
    
    def evaluate(self, state, ghost_belief, action):
        state = torch.cat([state, ghost_belief], dim=-1)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)        
        return action_logprobs, state_values, dist_entropy
    







class BeliefAwarePPO:
    def __init__(self, state_dim, ghost_action, action_dim, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        self.gamma = self.config['gamma']
        self.eps_clip = self.config['eps_clip']
        self.K_epochs = self.config['K_epochs']
        
        self.buffer = RolloutBuffer()

        self.policy = BeliefAwareActorCritic(state_dim, ghost_action, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': self.config['lr_actor']},
                        {'params': self.policy.critic.parameters(), 'lr': self.config['lr_critic']}
                    ])

        self.policy_old = BeliefAwareActorCritic(state_dim, ghost_action, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()


    def get_buffer_size(self):
        return len(self.buffer)

    def select_action(self, state, ghost_belief):

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            belief_probs = [F.softmax(logits, dim=-1) for logits in ghost_belief] 
            belief_vector = torch.cat(belief_probs, dim=-1).squeeze(0)
            if belief_vector.dim() == 1:
                belief_vector = belief_vector.unsqueeze(0) # [1, 20]

            action, action_logprob, state_val = self.policy_old.act(state, belief_vector)
            
            self.buffer.states.append(state)
            self.buffer.ghost_belief.append(belief_vector.detach().clone())
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            action = action.item()
            return action, action_logprob, state_val

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_ghost_belief = torch.squeeze(torch.stack(self.buffer.ghost_belief, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        # calculate advantages
        #logger.info(f"rewards shape: {rewards.shape}, old_state_values shape: {old_state_values.shape}")
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_ghost_belief, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path, filename="ppo_checkpoint.pth"):
        torch.save(self.policy_old.state_dict(), os.path.join(checkpoint_path, filename))
   
    def load(self, checkpoint_path, filename="ppo_checkpoint.pth"):
        file = os.path.join(checkpoint_path, filename)
        self.policy_old.load_state_dict(torch.load(file, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(file, map_location=lambda storage, loc: storage))









class BeliefAwareActorCriticAgent(nn.Module):
    def __init__(self, state_dim, ghost_dim, action_dim, config):
        super(BeliefAwareActorCriticAgent, self).__init__()
        self.config = config
        self.affine = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        logger.info("Affine network created")

        self.belief_net_dub = BeliefNet(config=self.config)

        self.ghost_state = nn.Sequential(
            nn.Linear(ghost_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
        logger.info("ghost state network created")
        
        self.action_layer = nn.Linear(128 + 128, action_dim)  # note the input size
        self.value_layer = nn.Linear(128 + 128, 1)
        
        self.logprobs = []
        self.state_values = []
        self.rewards = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, ghost_belief):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        belief_probs = [F.softmax(logits, dim=-1) for logits in ghost_belief] 
        belief_vector = torch.cat(belief_probs, dim=-1).squeeze(0)
        if belief_vector.dim() == 1:
            belief_vector = belief_vector.unsqueeze(0)  # [1, 20]

        # FIX: process state and belief separately
        state_feat = self.affine(state)           # [1, 128]
        ghost_feat = self.ghost_state(belief_vector) # [1, 128]

        combined = torch.cat([state_feat, ghost_feat], dim=-1)  # [1, 256]

        state_value = self.value_layer(combined)
        action_probs = F.softmax(self.action_layer(combined), dim=-1)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        
        pred_ghost_belief = self.belief_net_dub(state)
        target = torch.stack([probs.argmax(dim=-1) for probs in belief_probs], dim=1).to(state.device)  # [1, 4]

        # For each ghost, compute cross entropy loss
        aux_loss = 0
        for idx, pred_logits in enumerate(pred_ghost_belief):
            aux_loss += F.cross_entropy(pred_logits, target[:, idx])
        aux_loss /= len(pred_ghost_belief)


        return action.item(), aux_loss

    
    def calculateLoss(self, gamma=0.99):
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)
                
        # normalizing the rewards:
        rewards = torch.tensor(rewards).to(device=self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        
        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward  - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            loss += (action_loss + value_loss)   
        return loss
    
    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]


    def save(self, optimizer, path="src\\models\\ba_actor_critic", file="belief_aware_actor_critic.pth"):
        """
        Save the model and optimizer state dicts to the specified path.
        """
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, file)
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        logger.info(f"Model and optimizer saved to {save_path}")

    def load(self, optimizer, path="src\\models\\ba_actor_critic", file="belief_aware_actor_critic.pth", map_location=None):
        """
        Load the model and optimizer state dicts from the specified path.
        """
        load_path = os.path.join(path, file)
        if not os.path.exists(load_path):
            logger.error(f"Model file {load_path} does not exist.")
            return
        device = map_location if map_location is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(load_path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model and optimizer loaded from {load_path}")



