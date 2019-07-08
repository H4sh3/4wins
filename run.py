import gym
import gym_colonizer
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from collections import namedtuple
from itertools import count
import random
import sys

from agent.agent import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = Net().to(device)
target_net = Net().to(device)

def random_agent(episodes=10000):
	env = gym.make("Colonizer-v0")
	env.reset()
	env.render()

	for e in range(episodes):
		env.render()
		print(env.get_state())




#if __name__ == "__main__":
random_agent()
