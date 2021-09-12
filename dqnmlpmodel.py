import numpy as np
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.init import xavier_uniform

class DQNMLPnn(nn.Module):
  def __init__(self, input_shape, num_actions, hidden_dim):
    super().__init__()
    self.input_shape = input_shape
    self.num_actions = num_actions
    self.hidden_dim = hidden_dim
    
    
    self.fc1 = nn.Linear(self.input_shape, int(self.hidden_dim))
    # self.fc1_1 = nn.Linear(int(self.hidden_dim/2), int(self.hidden_dim/4))
    # self.fc1_2 = nn.Linear(int(self.hidden_dim/2), int(self.hidden_dim/4))
    self.fc2 = nn.Linear(int(self.hidden_dim), self.num_actions)
    
  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    # x = self.fc1_1(x)
    # x = F.relu(x)
    # x = self.fc1_2(x)
    # x = F.relu(x)
    x = self.fc2(x)
    return x

