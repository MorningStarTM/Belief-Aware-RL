from src.networks.belief_network import BeliefNet
from src.networks.belief_aware_rl import BeliefAwarePPO
from src.networks.ppo import PPO
from src.networks.trainer import BeliefAwareTrainer, PPOTrainer, ACTrainer
from src.networks.actor_critic import ActorCritic
from src.utils.logger import logger
import pickle
import os
from pac_man.pac_gym import PacManEnv
from src.utils.wrapper import RewardShapingWrapper
from src.utils.config import ppo_config, actor_config
import sys
import argparse



def ppo_trainer():
    env = PacManEnv()
    config = {
        'input_dim': 436,
        'hidden_dim': 64,
        'output_dim': 5,         # e.g., 5 actions per ghost
        'learning_rate': 1e-6,
        'epochs':10,
    }

    model = BeliefNet(config)
    model.load("src\\models\\belief_net.pth")
    logger.info("BeliefNet model initialized and loaded trained weights")

    wrapped_env = RewardShapingWrapper(env, belief_net=model)
    logger.info("Environment wrapped with RewardShapingWrapper using the BeliefNet model")  

    agent = PPO(state_dim=436, action_dim=5, config=ppo_config)
    logger.info("PPO agent initialized with the given configuration")

    trainer = PPOTrainer(agent=agent, env=wrapped_env, config=ppo_config)
    trainer.train()




def belief_aware_ppo_trainer():
    env = PacManEnv()
    config = {
        'input_dim': 436,
        'hidden_dim': 64,
        'output_dim': 5,         # e.g., 5 actions per ghost
        'learning_rate': 1e-6,
        'epochs':10,
    }

    model = BeliefNet(config)
    model.load("src\\models\\belief_net.pth")
    logger.info("BeliefNet model initialized and loaded trained weights")

    wrapped_env = RewardShapingWrapper(env, belief_net=model)
    logger.info("Environment wrapped with RewardShapingWrapper using the BeliefNet model")   

    agent = BeliefAwarePPO(state_dim=436, ghost_action=20, action_dim=5, config=ppo_config)
    logger.info("BeliefAwarePPO agent initialized with the given configuration")

    trainer = BeliefAwareTrainer(agent=agent, belief_model=model, env=wrapped_env, config=ppo_config)
    trainer.train()





def actor_critic_trainer():
    env = PacManEnv()
    config = {
        'input_dim': 436,
        'hidden_dim': 64,
        'output_dim': 5,         # e.g., 5 actions per ghost
        'learning_rate': 1e-6,
        'epochs':10,
        'episodes': 1,
        'max_ep_len':10000
    }

    model = BeliefNet(config)
    model.load("src\\models\\belief_net.pth")
    logger.info("BeliefNet model initialized and loaded trained weights")

    wrapped_env = RewardShapingWrapper(env, belief_net=model)
    logger.info("Environment wrapped with RewardShapingWrapper using the BeliefNet model")

    actor = ActorCritic(state_dim=436, action_dim=5, config=actor_config)
    logger.info("ActorCritic agent initialized with the given configuration")

    trainer = ACTrainer(model=actor, env=wrapped_env, config=actor_config)
    trainer.train()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Trainer Launcher")
    parser.add_argument("mode", type=str, choices=["ppo", "belief_aware_ppo", "actor_critic"],
                        help="Which trainer to run: ppo, belief_aware_ppo, or actor_critic")
    args = parser.parse_args()

    if args.mode == "ppo":
        ppo_trainer()
    elif args.mode == "belief_aware_ppo":
        belief_aware_ppo_trainer()
    elif args.mode == "actor_critic":
        actor_critic_trainer()