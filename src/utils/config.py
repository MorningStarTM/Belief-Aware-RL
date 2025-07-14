
actor_config = {
    'gamma' : 0.99,
    'lr' : 1e-4,
    'betas' : (0.9, 0.999),
    'random_seed' : 543,
    'render':False,
    'episodes': 1,
    'max_ep_len':10000
}


ppo_config = {
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