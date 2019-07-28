import gym
import gym_colonizer
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from itertools import count
import random
import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import math
from torch.autograd import Variable

from agent.agent import DQN, LSTM

use_cuda = torch.cuda.is_available
if use_cuda:
    print('using cuda!')
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

inputs = 129
n_actions = 54+72+1 # build village + build roads + doNothing
BUFFER_SIZE = 124

EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.1  # e-greedy threshold end value
EPS_DECAY = 100  # e-greedy threshold decay
GAMMA = 0.8  # Q-learning discount factor
LR = 0.1 
steps_done = 0

policy_net = LSTM(inputs,124,n_actions)
target_net = LSTM(inputs,124,n_actions)
if use_cuda:
    policy_net.cuda()
    target_net.cuda()

optimizer = optim.RMSprop(policy_net.parameters(),lr=LR)

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


def select_action(state,hidden,steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        out, hidden = policy_net(Variable(state),hidden)
        return out.type(FloatTensor),hidden#.data.max(1)[1].view(1, 1), hidden
    else:
        x = FloatTensor(n_actions).uniform_().unsqueeze(0)
        return x, None


def optimize_model(buffer, batch_size):
    if len(buffer) < batch_size:
        return buffer

    # random transition batch is taken from experience replay memory
    transitions = buffer.sample(batch_size)
    states, actions, next_states, rewards = zip(*transitions)
    states = Variable(torch.cat(states))
    actions = Variable(torch.cat(actions))
    rewards = Variable(torch.cat(rewards))
    next_states = Variable(torch.cat(next_states))

    policy_out, x = policy_net(states)
    state_action_values = policy_out.gather(1, actions)
    
    
    target_out, x = target_net(next_states)
    next_state_values = target_out.max(1)[0].view(batch_size, 1).detach()
    rewards = rewards.unsqueeze(1)
    expected_q_values = rewards + (GAMMA * next_state_values)
    loss = F.smooth_l1_loss(state_action_values, expected_q_values)

    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    env = gym.make("Colonizer-v0")
    env.reset()
    env.render()
    buffer = ReplayMemory(10000)
    iteration = 5000
    num_steps = 30
    rewards = []
    max_reward = 0
    median = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    Ln, = ax.plot(rewards)
    Ln2, = ax.plot(median)

    ax.set_ylim([0,100])
    plt.ylabel('Reward')
    plt.xlabel('Iteration')
    plt.ion()
    plt.show()    
    hidden = None
    for i in range(iteration):
        env.reset()
        rewards.append(0)
        reward = 0
        state = env.get_state()
        steps_done = 0
        for t in range(num_steps):
            env.render()
            action, h = select_action(FloatTensor([state]),hidden,i)
            if h:
                hidden = h
            
            action = env.filter_legal_actions(action,t).max(1).indices.unsqueeze(0)
            next_state, reward = env.step(action.item(),t)
            print(reward)
            rewards[i] += reward
            reward_t = torch.tensor([reward], device='cuda:0').view(1, 1)
            buffer.push((FloatTensor([state]),
                         action,
                         FloatTensor([next_state]),
                         FloatTensor([reward_t]))                        )
            # Replay memory
            optimize_model(buffer, BUFFER_SIZE)
            state = next_state
            median.append(np.mean(rewards))

        if rewards[i] > max_reward:
            max_reward = rewards[i]
            ax.set_ylim([0,max_reward+20])    

        if i % 20 == 0:
            ax.set_xlim([0,len(rewards)])

            Ln.set_ydata(rewards)
            Ln.set_xdata(range(len(rewards)))

            Ln2.set_ydata(median)
            Ln2.set_xdata(range(len(median)))

            plt.pause(0.00001)
    plt.savefig('plot.png')