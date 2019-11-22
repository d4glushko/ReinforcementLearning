import gym
from gym import wrappers
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .base_agent import BaseAgent


# hyper parameters
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
GAMMA = 0.9  # Q-learning discount factor
LR = 0.001  # NN optimizer learning rate
HIDDEN_LAYER = 256  # NN hidden layer size
BATCH_SIZE = 64  # Q-learning batch size
MEMORY_SIZE = 10000

# if gpu is to be used
# use_cuda = torch.cuda.is_available()
use_cuda = False
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Network(nn.Module):
    def __init__(self, observation_space: int, action_space: int):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(observation_space, HIDDEN_LAYER)
        self.l2 = nn.Linear(HIDDEN_LAYER, action_space)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

class TestAgent(BaseAgent):

    def __init__(self, observation_space: int, action_space: int):
        self.observation_space = observation_space
        self.action_space = action_space
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = ReplayMemory(MEMORY_SIZE)

        self.model = Network(observation_space, action_space)
        if use_cuda:
            self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), LR)

    def remember(self, state, action, reward, done, next_state):
        self.memory.push((FloatTensor([state]), LongTensor([[action]]), FloatTensor([next_state]), FloatTensor([reward])))

    def act(self, state):
        if random.random() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model(Variable(FloatTensor([state]), volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
        return q_values[0, 0]

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

        batch_state = Variable(torch.cat(batch_state))
        batch_action = Variable(torch.cat(batch_action))
        batch_reward = Variable(torch.cat(batch_reward))
        batch_next_state = Variable(torch.cat(batch_next_state))

        # current Q values are estimated by NN for all actions
        current_q_values = self.model(batch_state).gather(1, batch_action)
        # expected Q values are estimated from actions which gives maximum Q value
        max_next_q_values = self.model(batch_next_state).detach().max(1)[0]
        expected_q_values = batch_reward + (GAMMA * max_next_q_values)

        # loss is measured from error between current and newly expected Q values
        loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))

        # backpropagation of loss to NN
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
