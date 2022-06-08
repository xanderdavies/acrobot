import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from itertools import count
import random
import wandb
import math
from tqdm import tqdm 
import os

# display stuff
from gym import logger as gymlogger
# from gym.wrappers.monitoring import video_recorder
gymlogger.set_level(40) #error only
import glob
import io
import base64
from IPython.display import HTML
from IPython import display as ipythondisplay

from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

if not os.path.exists('./weights'):
  os.makedirs('./weights')

"""
Utility functions to enable video recording of gym environment and displaying it
To enable video, just do "env = wrap_env(env)""
"""

def show_video():
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else: 
    print("Could not find video")
    

# def wrap_env(env):
#   env = video_recorder.VideoRecorder(env, './video', enabled=True)
#   print(env)
#   return env

class PixelObs(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self.observations = deque([], maxlen=2)
  
  def reset(self):
    self.env.reset()
    self.observations.clear()
    self.observations.append(self._get_screen()) # previous observation
    return self._get_ob()

  def step(self, action):
    ob, reward, done, info = self.env.step(action)
    return self._get_ob(), reward, done, info

  def _get_screen(self):
    screen = torch.from_numpy(np.ascontiguousarray(env.render(mode="rgb_array")))
    return screen.permute(2, 0, 1).float()

  def _get_ob(self):
    self.observations.append(self._get_screen())
    return self.observations[1] - self.observations[0]

class DQN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 16, 5, 2)
    self.batch1 = nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(16, 32, 5, 2)
    self.batch2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 32, 5, 2)
    self.batch3 = nn.BatchNorm2d(32)
    self.head = nn.Linear(32*59*59, 3)
  
  def forward(self, x):
    x = F.relu(self.batch1(self.conv1(x)))
    x = F.relu(self.batch2(self.conv2(x)))
    x = F.relu(self.batch3(self.conv3(x)))
    return self.head(x.view(x.size(0), -1))

class ReplayBuffer():
  def __init__(self, capacity):
    self.memory = deque([], maxlen=capacity)

  def append(self, transition):
    self.memory.append(transition)

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)
  
  def __len__(self):
    return len(self.memory)

# get environment
env = gym.make("Acrobot-v1")
env = PixelObs(env)

NUM_EPS = 500 # Over 300
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
BATCH_SIZE = 64
TARGET_UPDATE = 10 # Number of episodes before updating Target network
GAMMA = 0.999
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"using {DEVICE}")

wandb.init(project="acrobot", tags=[f"bs_{BATCH_SIZE}", f"target_update_{TARGET_UPDATE}", f"gamma_{GAMMA}"], settings=wandb.Settings(start_method="thread"))

# get dqns
policy_model = DQN().to(DEVICE)
policy_model.load_state_dict(torch.load("weights/10.pt"))
target_model = DQN().to(DEVICE)
target_model.load_state_dict(policy_model.state_dict())

criterion = nn.SmoothL1Loss()
optimizer = torch.optim.RMSprop(policy_model.parameters())

def optimize_model(policy_model, target_model, replay_buffer):
  if len(replay_buffer) < BATCH_SIZE:
    return

  batch = replay_buffer.sample(BATCH_SIZE)
  state, action, reward, new_state = zip(*batch)
  reward_batch = torch.tensor(reward).unsqueeze(1).to(DEVICE)
  state_batch = torch.stack(state).to(DEVICE)
  action_batch = torch.tensor(action).unsqueeze(1).to(DEVICE)

  # q
  q = policy_model(state_batch).gather(1, action_batch)

  # non final states
  non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, new_state))).to(DEVICE) 
  non_final_new_states = torch.stack([s for s in new_state if s is not None]).to(DEVICE)

  # updated q
  v = target_model(non_final_new_states).max(1)[0].unsqueeze(1).to(DEVICE)
  new_q = reward_batch
  new_q[non_final_mask] += GAMMA * v

  loss = criterion(q, new_q)  
  optimizer.zero_grad()
  loss.backward()
  for param in policy_model.parameters():
        param.grad.data.clamp_(-1, 1)
  optimizer.step()
  wandb.log({"Loss": loss})

replay_buffer = ReplayBuffer(10000)
episode_lengths = []
steps_done = 0

for ep in tqdm(range(NUM_EPS)):
  obs = env.reset()

  for length in count():
    sample = np.random.rand()
    
    eps_thresh = eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_thresh:
      action = torch.argmax(policy_model(obs.unsqueeze(0).to(DEVICE))[0])
    
    else:
      action = env.action_space.sample() 
    
    steps_done += 1
    
    new_obs, reward, done, info = env.step(action) 
    if done:
      new_obs = None
    replay_buffer.append((obs, action, reward, new_obs))
    obs = new_obs

    optimize_model(policy_model, target_model, replay_buffer)
  
    if done:
      episode_lengths.append(length+1)
      wandb.log({"Episode": ep, "Episode Length": length+1})
      break
  if ep % TARGET_UPDATE == 0:
    torch.save(policy_model.state_dict(), f"weights/{ep}.pt")
  torch.save(policy_model.state_dict(), f"weights/policy.pt")