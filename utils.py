from collections import deque
import numpy as np
import torch
import gym
import random
from gym.wrappers.record_video import RecordVideo
import wandb
# from datetime import datetime

class PixelObs(gym.Wrapper):
    """
    Wraps around gym environment and returns pixel array as observation instead of (6,) vector.
    """

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
        screen = torch.from_numpy(np.ascontiguousarray(self.env.render(mode="rgb_array")))
        return screen.permute(2, 0, 1).float()

    def _get_ob(self):
        self.observations.append(self._get_screen())
        return self.observations[1] - self.observations[0]

class ReplayBuffer():
    """
    Memory class for storing transitions.
    """
    
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

def optimize_model(policy_model, target_model, replay_buffer, criterion, optimizer, batch_size, gamma, device):
    if len(replay_buffer) < batch_size:
        return

    # then = datetime.now()
    batch = replay_buffer.sample(batch_size)
    state, action, reward, new_state = zip(*batch)
    # print("Sampled batch in {}".format((datetime.now() - then)/1000))

    # then = datetime.now()
    reward_batch = torch.tensor(reward).unsqueeze(1).to(device)
    state_batch = torch.stack(state).to(device)
    action_batch = torch.tensor(action).unsqueeze(1).to(device)
    # print("setup time:", (datetime.now() - then).microseconds/1000)

    # q
    # then = datetime.now()
    q = policy_model(state_batch).gather(1, action_batch)
    # print("q time:", (datetime.now() - then).microseconds/1000)

    # non final states
    # then = datetime.now()
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, new_state))).to(device) 
    non_final_new_states = torch.stack([s for s in new_state if s is not None]).to(device)
    # print("non final states time:", (datetime.now() - then).microseconds/1000)

    # updated q
    # then = datetime.now()
    v = target_model(non_final_new_states).max(1)[0].unsqueeze(1).to(device)
    new_q = reward_batch
    new_q[non_final_mask] += gamma * v
    # print("updated q time:", (datetime.now() - then).microseconds/1000)

    # loss
    # then = datetime.now()
    loss = criterion(q, new_q)  
    optimizer.zero_grad()
    loss.backward()
    for param in policy_model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    wandb.log({"Loss": loss})
    # print("loss time:", (datetime.now() - then).microseconds/1000)

def save_example_video(env, policy_model, path):
    """
    Saves a video of the environment.
    """
    env = RecordVideo(env, path)
    env.reset()
    done = False
    while not done:
        action = policy_model(env._get_ob()).argmax().item()
        ob, reward, done, info = env.step(action)
    env.close()