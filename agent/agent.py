import random
import math
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import *
from torch.autograd import Variable

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
        #super(DQN, self).__init__()
        nn.Module.__init__(self)
        hidden_layer = 120
        self.model = nn.Sequential(nn.RNN(input_layer,3,hidden_layer),nn.Linear(hidden_layer, outputs))

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        out, y = self.model(x)
        return out

class LSTM(nn.Module):
    def __init__(self, inputs,hidden_size,outputs):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size

        self.inp = nn.Linear(inputs, hidden_size)#.to("cuda")
        self.rnn = nn.LSTM(hidden_size, hidden_size, 2, batch_first=False)#.to("cuda")
        self.out = nn.Linear(hidden_size, outputs)#.to("cuda")
        self.relu = nn.ReLU()#.to("cuda")

    def step(self, input, hidden=None):
        # input = self.inp(input.view(1, -1)).unsqueeze(1)
        input = self.inp(input).unsqueeze(0).unsqueeze(0)
        if len(input.shape) == 4:
            input = input[0]
        output, hidden = self.rnn(input, hidden)
        output = self.out(output)#.squeeze(1))
        output = self.relu(output)
        return output[0], hidden

    def forward(self, inputs, hidden=None, force=True, steps=0):
        output, hidden = self.step(inputs, hidden)
        return output, hidden

