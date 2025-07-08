from src.utils.dataset import PacmanGhostActionDataset
from torch.utils.data import DataLoader
from src.networks.belief_network import BeliefNet
from src.networks.trainer import Trainer
from src.utils.logger import logger
import pickle




# Load the data
with open('src\\data\\ppo_rollout_data.pkl', 'rb') as f:
    data = pickle.load(f)

logger.info("Train Data loaded successfully from ppo_rollout_data.pkl")

# To extract all states and actions into flat lists:
all_states = []
all_ghost_actions = []

for episode in data:
    for step in episode:
        all_states.append(step["state"])
        all_ghost_actions.append(step["ghosts_last_action"])

train_dataset = PacmanGhostActionDataset(data)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
logger.info("Train DataLoader created with batch size 8 and shuffling enabled")

############################################################################################################
with open('src\\data\\test.pkl', 'rb') as f:
    data = pickle.load(f)
logger.info("Test Data loaded successfully from ppo_rollout_data.pkl")
# To extract all states and actions into flat lists:
all_states = []
all_ghost_actions = []

for episode in data:
    for step in episode:
        all_states.append(step["state"])
        all_ghost_actions.append(step["ghosts_last_action"])

test_dataset = PacmanGhostActionDataset(data)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)
logger.info("Test DataLoader created with batch size 8 and shuffling enabled")



config = {
    'input_dim': len(all_states[0]),
    'hidden_dim': 64,
    'output_dim': 5,         # e.g., 5 actions per ghost
    'learning_rate': 1e-6,
    'epochs':5,
}



model = BeliefNet(config)
logger.info("BeliefNet model initialized with")

trainer = Trainer(model, train_loader, test_loader, config)
trainer.fit()