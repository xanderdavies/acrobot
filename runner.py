import gym
import torch
import torch.nn as nn
import numpy as np
from itertools import count
import wandb
from tqdm import tqdm 
from utils import *
from dqn import DQN
from pyvirtualdisplay import Display
from datetime import datetime

display = Display(visible=0, size=(1400, 900))
display.start()

NUM_EPS = 400 # Over 300
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
BATCH_SIZE = 64
TARGET_UPDATE = 10 # Number of episodes before updating Target network
GAMMA = 0.999
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LOAD_PATH = "weights/latest_1.pt" # None
print(f"using {DEVICE}")

wandb.init(project="acrobot", tags=[f"bs_{BATCH_SIZE}"])

# get dqns
policy_model = DQN().to(DEVICE)
policy_model.train()
if LOAD_PATH is not None:
    policy_model.load_state_dict(torch.load(LOAD_PATH))
target_model = DQN().to(DEVICE)
target_model.load_state_dict(policy_model.state_dict());
target_model.eval()

# get env
env = gym.make("Acrobot-v1")
env = PixelObs(env)

# config
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.RMSprop(policy_model.parameters())
replay_buffer = ReplayBuffer(10000)
episode_lengths = []

# train
for ep in tqdm(range(NUM_EPS)):
    obs = env.reset()

    for length in tqdm(count(), total=500, leave=False, desc=f"Last episode length was {episode_lengths[-1] if len(episode_lengths) > 0 else 'NA'}"):
        sample = np.random.rand()
        
        eps_thresh = EPS_END + (EPS_START - EPS_END) * np.exp(-ep/EPS_DECAY)

        if sample > eps_thresh:
            with torch.no_grad():
                action = torch.argmax(policy_model(obs.unsqueeze(0).to(DEVICE))[0])
        else:
            action = env.action_space.sample() 
        
        new_obs, reward, done, info = env.step(action) 

        if done:
            new_obs = None
        replay_buffer.append((obs, action, reward, new_obs))
        obs = new_obs

        optimize_model(policy_model, target_model, replay_buffer, criterion, optimizer, BATCH_SIZE, GAMMA, DEVICE)

        if done:
            episode_lengths.append(length+1)
            wandb.log({"Episode": ep, "Episode Length": length+1})
            break

    if ep % TARGET_UPDATE == 0:
        target_model.load_state_dict(policy_model.state_dict())
    torch.save(policy_model.state_dict(), "latest.pt")
    print(f"Last episode length {episode_lengths[-1]}")