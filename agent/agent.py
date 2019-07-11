import random
import math
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import *


# Code taken from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, input_layer, outputs):
        super(DQN, self).__init__()
        hidden_layer = 256
        #self.model = nn.Sequential(nn.Linear(input_layer, hidden_layer),
        #             nn.ReLU(),
        #             nn.Linear(hidden_layer, outputs),
        #             nn.Sigmoid())
        self.model = nn.Sequential(nn.Linear(input_layer, hidden_layer),
                     nn.ReLU(),
                     nn.Linear(hidden_layer, outputs))

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        return self.model(x)

def select_action(device, state, n_actions, steps_done, policy_net):
    sample = np.random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor(
            [[random.randrange(n_actions)]], device=device, dtype=torch.long
        )
