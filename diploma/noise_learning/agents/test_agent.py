import typing
import gym
from gym import wrappers
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import namedtuple

from .base_agent import BaseAgent


# hyper parameters
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
GAMMA = 1  # Q-learning discount factor
LR = 0.001  # NN optimizer learning rate
HIDDEN_LAYER = 24  # NN hidden layer size
BATCH_SIZE = 32  # Q-learning batch size
MEMORY_SIZE = 50000

# if gpu is to be used
use_cuda = torch.cuda.is_available()
# use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
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

class Network(nn.Module):
    def __init__(self, observation_space: int, action_space: int):
        super(Network, self).__init__()
        self.l1 = nn.Linear(observation_space, HIDDEN_LAYER)
        self.l2 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.l3 = nn.Linear(HIDDEN_LAYER, action_space)

        # Manual initialization of weights and biases
        # with torch.no_grad():
        #     self.l1.weight = nn.Parameter(torch.tensor(l1w, device=device, dtype=torch.float))
        #     self.l2.weight = nn.Parameter(torch.tensor(l2w, device=device, dtype=torch.float))
        #     self.l3.weight = nn.Parameter(torch.tensor(l3w, device=device, dtype=torch.float))
        #     self.l1.bias = nn.Parameter(torch.tensor(l1b, device=device, dtype=torch.float))
        #     self.l2.bias = nn.Parameter(torch.tensor(l2b, device=device, dtype=torch.float))
        #     self.l3.bias = nn.Parameter(torch.tensor(l3b, device=device, dtype=torch.float))

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class TestAgent(BaseAgent):
    def __init__(self, observation_space: int, action_space: int, debug: bool):
        super().__init__(observation_space, action_space, debug)
        self.exploration_rate = EXPLORATION_MAX
        self.memory = ReplayMemory(MEMORY_SIZE)

        self.model = Network(observation_space, action_space).to(device)

        # Add debug hooks
        # def printdata(self, input, output):
        #     print(f"Inside {self.__class__.__name__} forward")
        #     print(f"Input data: {input}")
        #     print(f"Output data: {output.data}")

        # def printgraddata(self, input, output):
        #     print(f"Inside {self.__class__.__name__} backward")
        #     print(f"Grad Input data: {input[0]}")
        #     print(f"Grad Output data: {output[0]}")
        # self.model.register_forward_hook(printdata)
        # self.model.register_backward_hook(printgraddata)

        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

    def remember(self, state, action, reward, done, next_state):
        super().remember(state, action, reward, done, next_state)
        state = torch.tensor([state], device=device, dtype=torch.float)
        action = torch.tensor([[action]], device=device, dtype=torch.long)
        if next_state is not None:
            next_state = torch.tensor([next_state], device=device, dtype=torch.float)
        reward = torch.tensor([reward], device=device, dtype=torch.float)
        self.memory.push(state, action, next_state, reward)

    def act(self, state):
        super().act(state)
        if random.random() < self.exploration_rate:
            return random.randrange(self.action_space)
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return self.model(torch.tensor([state], device=device, dtype=torch.float)).max(1)[1].item()

    def reflect(self):
        if len(self.memory) < BATCH_SIZE:
            return

        super().reflect()

        loss = self.__get_loss(self.memory.sample(BATCH_SIZE))
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
        self.last_loss = loss.item()

    def __get_loss(self, transitions: typing.List[Transition]):
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the model; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.model(non_final_next_states).detach().max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute MSE Loss
        loss = nn.MSELoss()
        return loss(state_action_values, expected_state_action_values.unsqueeze(1))
