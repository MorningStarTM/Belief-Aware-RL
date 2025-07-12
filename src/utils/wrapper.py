import gym
import numpy as np
from src.networks.belief_network import BeliefNet


class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env, belief_net:BeliefNet, vision_radius=3):
        super().__init__(env)
        self.belief_net = belief_net
        self.vision_radius = vision_radius

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs_to_state(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        pacman_pos = obs['pacman']['pos']
        ghosts_obs = obs['ghosts']

        # Flatten obs for BeliefNet
        flat_obs = obs_to_state(obs)  # shape [N,]
        belief_pred_actions = self.belief_net.predict_actions(flat_obs)[0].cpu().numpy()  # shape: (4,)

        # Only penalize if Pac-Man is close and BeliefNet's prediction matches reality
        for i, ghost in enumerate(ghosts_obs):
            ghost_row, ghost_col = ghost[0], ghost[1]
            if ghost_row >= 0:  # visible ghost
                dist = abs(pacman_pos[0] - ghost_row) + abs(pacman_pos[1] - ghost_col)
                real_action = self.env.last_ghost_actions[i]
                pred_action = belief_pred_actions[i]
                if dist <= self.vision_radius:
                    if pred_action == real_action:
                        if dist == 1:
                            # Pac-Man is next to a ghost, and should have known
                            reward -= 5  # harsh penalty
                        if pacman_pos[0] == ghost_row and pacman_pos[1] == ghost_col:
                            # Pac-Man collides with ghost, and should have known
                            reward -= 15  # even harsher penalty
        
        # 4. Check collision
        for ghost in self.ghosts:
            if self.pacman.rect.colliderect(ghost.rect):
                reward -= 10
                self.done = True
                break

        # 5. Check win
        if all(not coin for row in self.coins for coin in row):
            reward += 10
            self.done = True

        return obs_to_state(obs), reward, done, info



def obs_to_state(obs):
        # Example: flatten all obs into 1D numpy array (adjust if your model expects differently)
        state = []
        state.extend(obs['pacman']['pos'].tolist())
        state.extend(obs['pacman']['direction'].tolist())
        state.append(obs['pacman']['last_action'])
        state.extend(obs['ghosts'].flatten().tolist())
        state.extend(obs['coins'].tolist())
        return state