from src.networks.ppo import PPO
from src.networks.actor_critic import ActorCritic
from src.networks.belief_aware_rl import BeliefAwarePPO
from src.utils.logger import logger
import os
import numpy as np
from datetime import datetime
import torch
from src.networks.belief_network import BeliefNet
from torch.utils.tensorboard import SummaryWriter
import os



class PPOTrainer:
    def __init__(self, agent:PPO, env, config):
        
        self.env = env
        self.env_name = "ToMPacMan"
        self.agent = agent
        self.best_score = 0.0
        self.score_history = []
        self.config = config
        self.episode_rewards = []  # Stores total reward per episode
        self.step_rewards = []     # Stores every single reward at each timestep

        self.log_dir = "model_logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.log_dir = self.log_dir + '/' + self.env_name + '/'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        run_num = 0
        current_num_files = next(os.walk(self.log_dir))[2]
        run_num = len(current_num_files)

        #### create new log file for each run
        self.log_f_name = self.log_dir + '/PPO_' + self.env_name + "_log_" + str(run_num) + ".csv"

        logger.info("current logging run number for " + self.env_name + " : ", run_num)
        logger.info("logging at : " + self.log_f_name)

        self.directory = "src\\models"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.directory = self.directory + '/' + 'ppo' + '/'
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        logger.info(f"directory for saving model weights : {self.directory}")

        self.reward_folder = 'src\\rewards'
        if not os.path.exists(self.reward_folder):
            os.makedirs(self.reward_folder)

        self.reward_folder = self.reward_folder + '/' + 'ppo' + '/'
        if not os.path.exists(self.reward_folder):
            os.makedirs(self.reward_folder)
        logger.info(f"directory for saving rewards : { self.reward_folder}")


    def train(self):
        start_time = datetime.now().replace(microsecond=0)
        logger.info("Started training at (GMT) : ", start_time)

        print("============================================================================================")

        # logging file
        log_f = open(self.log_f_name,"w+")
        log_f.write('episode,timestep,reward\n')

        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0

        time_step = 0
        i_episode = 0
        
        while time_step <= self.config['max_training_timesteps']:

            state, _ = self.env.reset()
            current_ep_reward = 0

            for t in range(1, self.config['max_ep_len']+1):

                # select action with policy

                action, *_ = self.agent.select_action(state)
                state, reward, done, _, _ = self.env.step(action)
                self.step_rewards.append(reward)

                # saving reward and is_terminals
                self.agent.buffer.rewards.append(reward)
                self.agent.buffer.is_terminals.append(done)

                time_step +=1
                current_ep_reward += reward

                # update PPO agent
                if time_step % self.config['update_timestep'] == 0:
                    self.agent.update()


                # log in logging file
                if time_step % self.config['log_freq'] == 0:

                    # log average reward till last episode
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_avg_reward = round(log_avg_reward, 4)

                    log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                    log_f.flush()

                    log_running_reward = 0
                    log_running_episodes = 0

                # printing average reward
                if time_step % self.config['print_freq'] == 0:

                    # print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)

                    logger.info("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                    print_running_reward = 0
                    print_running_episodes = 0

                # save model weights
                if time_step % self.config['save_model_freq'] == 0:
                    logger.info("--------------------------------------------------------------------------------------------")
                    logger.info("saving model at : " + self.directory)
                    self.agent.save(self.directory)
                    logger.info("model saved")
                    logger.info("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    logger.info("--------------------------------------------------------------------------------------------")

                # break; if the episode is over
                if done:
                    break
            
            self.episode_rewards.append(current_ep_reward)  
            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            i_episode += 1

        log_f.close()
        self.env.close()

        # print total training time
        logger.info("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        logger.info("Started training at (GMT) : ", start_time)
        logger.info("Finished training at (GMT) : ", end_time)
        logger.info("Total training time  : ", end_time - start_time)
        logger.info("============================================================================================")

        np.save(os.path.join(self.reward_folder, f"ppo_{self.env_name}_step_rewards.npy"), np.array(self.step_rewards))
        np.save(os.path.join(self.reward_folder, f"ppo_{self.env_name}_episode_rewards.npy"), np.array(self.episode_rewards))
        logger.info(f"Saved step_rewards and episode_rewards to {self.log_dir}")





##################################################################################################################################################################
class BeliefAwareTrainer:
    def __init__(self, agent:BeliefAwarePPO, belief_model:BeliefNet, env, config):
        
        self.env = env
        self.env_name = "ToMPacMan"
        self.agent = agent
        self.belief_model = belief_model
        self.best_score = 0.0
        self.score_history = []
        self.config = config
        self.episode_rewards = []  # Stores total reward per episode
        self.step_rewards = []     # Stores every single reward at each timestep

        self.log_dir = "model_logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.log_dir = self.log_dir + '/' + self.env_name + '/'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        run_num = 0
        current_num_files = next(os.walk(self.log_dir))[2]
        run_num = len(current_num_files)

        #### create new log file for each run
        self.log_f_name = self.log_dir + '/BAPPO_' + self.env_name + "_log_" + str(run_num) + ".csv"

        logger.info("current logging run number for " + self.env_name + " : ", run_num)
        logger.info("logging at : " + self.log_f_name)

        self.directory = "src\\models"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.directory = self.directory + '/' + 'BeliefAwarePPO' + '/'
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        logger.info("directory for saving model weights : ", self.directory)


        self.reward_folder = 'src\\rewards'
        if not os.path.exists(self.reward_folder):
            os.makedirs(self.reward_folder)

        self.reward_folder = self.reward_folder + '/' + 'BeliefAwarePPO' + '/'
        if not os.path.exists(self.reward_folder):
            os.makedirs(self.reward_folder)
        logger.info("directory for saving rewards : ", self.reward_folder)



    def train(self):
        start_time = datetime.now().replace(microsecond=0)
        logger.info("Started training at (GMT) : ", start_time)

        logger.info("============================================================================================")

        # logging file
        log_f = open(self.log_f_name,"w+")
        log_f.write('episode,timestep,reward\n')

        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0

        time_step = 0
        i_episode = 0
        
        while time_step <= self.config['max_training_timesteps']:

            state, _ = self.env.reset()
            current_ep_reward = 0

            for t in range(1, self.config['max_ep_len']+1):

                # select action with policy
                #obs = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.belief_model.device) #make tensor for belief model
                #ghost_belief = self.belief_model(obs)
                ghost_belief = torch.rand(1, 20).to(self.belief_model.device)

                action, *_ = self.agent.select_action(state, ghost_belief)
                state, reward, done, _, _ = self.env.step(action)
                self.step_rewards.append(reward)

                # saving reward and is_terminals
                self.agent.buffer.rewards.append(reward)
                self.agent.buffer.is_terminals.append(done)

                time_step +=1
                current_ep_reward += reward

                # update PPO agent
                if time_step % self.config['update_timestep'] == 0:
                    self.agent.update()


                # log in logging file
                if time_step % self.config['log_freq'] == 0:

                    # log average reward till last episode
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_avg_reward = round(log_avg_reward, 4)

                    log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                    log_f.flush()

                    log_running_reward = 0
                    log_running_episodes = 0

                # printing average reward
                if time_step % self.config['print_freq'] == 0:

                    # print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)

                    logger.info("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                    print_running_reward = 0
                    print_running_episodes = 0

                # save model weights
                if time_step % self.config['save_model_freq'] == 0:
                    logger.info("--------------------------------------------------------------------------------------------")
                    logger.info("saving model at : " + self.directory)
                    self.agent.save(self.directory)
                    logger.info("model saved")
                    logger.info("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    logger.info("--------------------------------------------------------------------------------------------")

                # break; if the episode is over
                if done:
                    break
            
            self.episode_rewards.append(current_ep_reward)  
            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            i_episode += 1

        log_f.close()
        self.env.close()

        # print total training time
        logger.info("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        logger.info("Started training at (GMT) : ", start_time)
        logger.info("Finished training at (GMT) : ", end_time)
        logger.info("Total training time  : ", end_time - start_time)
        logger.info("============================================================================================")

        np.save(os.path.join(self.reward_folder, f"belief_aware_ppo_with_random_step_rewards.npy"), np.array(self.step_rewards))
        np.save(os.path.join(self.reward_folder, f"belief_aware_ppo_with_random_episode_rewards.npy"), np.array(self.episode_rewards))
        logger.info(f"Saved step_rewards and episode_rewards to {self.log_dir}")









class ACTrainer:
    def __init__(self, model:ActorCritic, env, config):
        self.config = config
        torch.manual_seed(config['random_seed'])
        self.model = model
        self.env = env
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=config['betas'])


    def train(self):
        running_reward = 0
        for i_episode in range(0, 10000):
            logger.info(f"Episode : {i_episode}")
            state, _ = self.env.reset()
            for t in range(10000):
                action = self.model(state)
                state, reward, done, _,_ = self.env.step(action)
                self.model.rewards.append(reward)
                running_reward += reward
                if self.config['render'] and i_episode > 1000:
                    self.env.render()
                if done:
                    break
            
            # Updating the policy :
            self.optimizer.zero_grad()
            loss = self.model.calculateLoss(self.config['gamma'])
            loss.backward()
            self.optimizer.step()        
            self.model.clearMemory()
            
            # saving the model if episodes > 999 OR avg reward > 200 
            if i_episode % 1000 == 0:
                self.model.save()    
           
            
            if i_episode % 20 == 0:
                running_reward = running_reward/20
                logger.info('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, running_reward))
                running_reward = 0

        



####################################################################################################################################
class Trainer:
    def __init__(self, model:BeliefNet, train_loader, val_loader, config, log_dir="runs/beliefnet"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = model.optimizer
        self.epochs = config['epochs']
        self.best_val_loss = float('inf')
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

        self.model_save_path = 'src\\models'
        self.model_save_path = os.path.join(self.model_save_path, 'belief_net.pth')


    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total = 0
        for states, targets in self.train_loader:
            states = states.to(self.device)
            targets = targets.to(self.device)  # shape: [batch, 4]

            outputs = self.model(states)       # list of 4 [batch, n_actions]
            loss = 0
            for i in range(4):
                loss += self.model.loss_fn(outputs[i], targets[:, i])
            loss = loss / 4

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * states.size(0)
            total += states.size(0)

            # Log training loss for every batch
            self.writer.add_scalar("Loss/Train_batch", loss.item(), self.global_step)
            self.global_step += 1

        avg_loss = total_loss / total
        self.writer.add_scalar("Loss/Train_epoch", avg_loss, epoch)
        return avg_loss

    def evaluate(self, epoch):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for states, targets in self.val_loader:
                states = states.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(states)
                loss = 0
                batch_correct = 0

                for i in range(4):
                    loss += self.model.loss_fn(outputs[i], targets[:, i])
                    preds = outputs[i].argmax(dim=1)
                    batch_correct += (preds == targets[:, i]).sum().item()

                loss = loss / 4
                total_loss += loss.item() * states.size(0)
                total_correct += batch_correct
                total_samples += states.size(0) * 4  # 4 heads per sample

        avg_loss = total_loss / total_samples * 4  # because we divide by (N*4)
        accuracy = total_correct / total_samples
        self.writer.add_scalar("Loss/Val_epoch", avg_loss, epoch)
        self.writer.add_scalar("Accuracy/Val_epoch", accuracy, epoch)
        return avg_loss, accuracy

    def fit(self):
        logger.info("Starting training for BeliefNet...")
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_acc = self.evaluate(epoch)
            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_acc:.4f}")

            # Save the model if the validation loss decreased
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.model.save(self.model_save_path)
                logger.info(f"Saved new best model at epoch {epoch} with val_loss={val_loss:.4f}")

        self.writer.close()
