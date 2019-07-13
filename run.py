import gym
import gym_colonizer
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from collections import namedtuple
from itertools import count
import random
import sys
import time
import matplotlib
import matplotlib.pyplot as plt
from torch.autograd import Variable

from agent.agent import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

n_actions = 54
GAMMA = 0.8

policy_net = DQN(73, n_actions).to(device)
target_net = DQN(73, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())

plt.ion()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def select_action(state, eps=0.10):
    rand = random.uniform(0, 1)
    if rand > eps:
        with torch.no_grad():
            return policy_net(state).type(torch.FloatTensor)
            
    else:
        return torch.rand([n_actions])


def optimize_model(buffer, batch_size, gamma=0.999):
    if len(buffer) < batch_size:
        return

    # random transition batch is taken from experience replay memory
    transitions = buffer.sample(batch_size)
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

    batch_state = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))

    batch_reward = Variable(torch.cat(batch_reward))
    batch_next_state = Variable(torch.cat(batch_next_state))

    # current Q values are estimated by NN for all actions
    current_q_values = policy_net(batch_state).gather(1, batch_action.long())
    # expected Q values are estimated from actions which gives maximum Q value
    max_next_q_values = policy_net(batch_next_state).detach().max(1)[0]
    expected_q_values = batch_reward + (GAMMA * max_next_q_values)
    # loss is measured from error between current and newly expected Q values

    #loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze_(1))
    loss = F.smooth_l1_loss(current_q_values, expected_q_values)

    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    env = gym.make("Colonizer-v0")
    env.reset()
    env.render()
    buffer = ReplayMemory(10000)
    iteration = 400
    num_steps = 8
    rewards = []

    fig = plt.figure()
    ax = fig.add_subplot(111)
    Ln, = ax.plot(rewards)
    ax.set_ylim([0,100])
    plt.ylabel('Reward')
    plt.xlabel('Iteration')
    plt.ion()
    plt.show()    

    for i in range(iteration):
        env.reset()
        rewards.append(0)
        reward = 0
        for t in range(num_steps):
            env.render()
            state = env.get_state()
            state = torch.FloatTensor([state])

            last_state = state
            action = select_action(state[0])
            action = action.max(0).indices
            
            reward = env.step(action.item())
            rewards[i] += reward

            reward_t = torch.FloatTensor([reward]).view(1, 1)
            
            next_state = torch.FloatTensor([env.get_state()]) - last_state
            buffer.push((state, torch.LongTensor([[action]]), next_state, reward_t))
            # Replay memory
            optimize_model(buffer, 128)
        ax.set_xlim([0,len(rewards)])
        Ln.set_ydata(rewards)
        Ln.set_xdata(range(len(rewards)))
        plt.pause(0.0001)
    plt.savefig('plot.png')