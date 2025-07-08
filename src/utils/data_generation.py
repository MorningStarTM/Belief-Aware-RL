from pac_man.pac_gym import FlattenObsWrapper, obs_to_state, PacManEnv
from src.networks.ppo import PPO
from src.utils.logger import logger

config = {
        'gamma': 0.99,
        'eps_clip': 0.2,
        'K_epochs': 4,
        'lr_actor': 1e-4,
        'lr_critic': 1e-3,
        'max_training_timesteps': int(3e6),  # Adjust as needed
        'max_ep_len': 1000,                  # Max timesteps per episode
        'update_timestep': 1000 * 4,            # How often to update PPO
        'log_freq': 1000 * 2,                   # How often to log
        'print_freq': 2000 * 5,                 # How often to print stats
        'save_model_freq': int(1e5)           # How often to save model
    }

def data_collect(n_episodes = 500):
    import pickle

    env = PacManEnv()
    wrapped_env = FlattenObsWrapper(env)
    state_dim = len(obs_to_state(env.reset()))
    action_dim = env.action_space.n

    agent = PPO(state_dim, action_dim, config)
    agent.load("src\\models\\ppo")  # Path where your model was saved

      # Number of rollouts to collect
    data = []

    for ep in range(n_episodes):
        logger.info(f"episode : {ep}")
        obs = env.reset()
        state = obs_to_state(obs)
        done = False
        episode_data = []
        while not done:
            # 1. Collect state and *current* ghost actions BEFORE action is taken
            # This is the crucial fix:
            ghost_actions = getattr(env, "last_ghost_actions", None)
            sample = {
                "state": state.copy(),
                "ghosts_last_action": ghost_actions.copy() if ghost_actions is not None else None
            }
            episode_data.append(sample)

            # 2. Agent acts
            action, *_ = agent.select_action(state)
            obs, reward, done, info = env.step(action)

            # 3. Next state for next step
            state = obs_to_state(obs)
        data.append(episode_data)
    
    # Save to disk (optional)
    with open('src\\data\\test.pkl', 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    data_collect()
    print("Data collection complete.")