import numpy as np
import random
import torch
from collections import deque, namedtuple

class ReplayMemoryBuffer:
  def __init__(self, buffer_size):
    self.buffer = deque(maxlen=buffer_size)
    self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "legal_actions_mask"])
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  def add(self, state_p, action_p, reward_p, next_state_p, done_p, legal_actions_mask_p):
    e_p = self.experience(state_p, action_p, reward_p, next_state_p, done_p, legal_actions_mask_p) # 这里传进来的是地址，如果一直传一样的就会得到一个列表with一样的值
    self.buffer.append(e_p)
    return e_p
    
  def sample(self, batch_size):
    self.buffer_s = len(self.buffer)
    assert self.buffer_s > batch_size, "The buffer does not contain enough memories for training."
    
    sampled_experiences = random.sample(self.buffer, k=batch_size)
    
    # print(sampled_experiences)
    # exit(1)
    
    states = torch.from_numpy(np.array([e.state for e in sampled_experiences if e != None])).float().to(self.device)
    # print(type(states))
    # exit(1)
    actions = torch.from_numpy(np.array([e.action for e in sampled_experiences if e != None])).float().to(self.device)
    rewards = torch.from_numpy(np.array([e.reward for e in sampled_experiences if e != None])).float().to(self.device)
    next_states = torch.from_numpy(np.array([e.next_state for e in sampled_experiences if e != None])).float().to(self.device)
    dones = torch.from_numpy(np.array([e.done for e in sampled_experiences if e != None])).float().to(self.device)
    legal_actions_mask = torch.from_numpy(np.array([e.legal_actions_mask for e in sampled_experiences if e!=None])).float().to(self.device)
        
    return (states, actions, rewards, next_states, dones, legal_actions_mask)

  def __len__(self):
    return len(self.buffer)
