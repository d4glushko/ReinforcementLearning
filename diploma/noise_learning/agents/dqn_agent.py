import typing
import gym
from gym import wrappers
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import namedtuple
import os

from .base_agent import BaseAgent, AgentHyperParams


# hyper parameters
EXPLORATION_MAX = 0.99
EXPLORATION_MIN = 0.02
EXPLORATION_DECAY = 0.9996
GAMMA = 1.0  # Q-learning discount factor
LR = 1e-3 # NN optimizer learning rate
HIDDEN_LAYERS_SIZES = [256]  # NN hidden layer size
BATCH_SIZE = 32 # Q-learning batch size

MEMORY_SIZE = 50000
UPD_TRG_MODEL = 1000
DOUBLE_DQN = True

DROPOUT_P = 0.0

FRAMES_TO_CONSIDER = 4

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DqnAgentHyperParams(AgentHyperParams):
    def __init__(self):
        self.learning_rate: float = LR
        self.gamma: float = GAMMA
        self.hidden_layers_sizes: typing.List[int] = HIDDEN_LAYERS_SIZES
        self.batch_size: int = BATCH_SIZE
        self.memory_size: int = MEMORY_SIZE#[current_execution]
        self.exploration_max: float = EXPLORATION_MAX
        self.exploration_min: float = EXPLORATION_MIN
        self.exploration_decay: float = EXPLORATION_DECAY
        self.update_trg_model: int = UPD_TRG_MODEL
        self.double_dqn: bool = DOUBLE_DQN
        self.dropout_p: float = DROPOUT_P
        self.frames_to_consider: int = FRAMES_TO_CONSIDER

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
    def __init__(self, observation_space: tuple, action_space: int, hidden_layers_sizes: typing.List[int], dropout_p: float):
        super(Network, self).__init__()
        self.l1 = nn.Linear(observation_space[0], hidden_layers_sizes[0])
        self.dr = nn.Dropout(p=dropout_p)
        # self.l2 = nn.Linear(hidden_layers_sizes[0], hidden_layers_sizes[1])
        self.l3 = nn.Linear(hidden_layers_sizes[0], action_space)

        # Manual initialization of weights and biases
        # with torch.no_grad():
        #     self.l1.weight = nn.Parameter(torch.tensor(l1w, dtype=torch.float).to(self.device, non_blocking=True))
        #     self.l2.weight = nn.Parameter(torch.tensor(l2w, dtype=torch.float).to(self.device, non_blocking=True))
        #     self.l3.weight = nn.Parameter(torch.tensor(l3w, dtype=torch.float).to(self.device, non_blocking=True))
        #     self.l1.bias = nn.Parameter(torch.tensor(l1b, dtype=torch.float).to(self.device, non_blocking=True))
        #     self.l2.bias = nn.Parameter(torch.tensor(l2b, dtype=torch.float).to(self.device, non_blocking=True))
        #     self.l3.bias = nn.Parameter(torch.tensor(l3b, dtype=torch.float).to(self.device, non_blocking=True))

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.dr(x)
        # x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

    def get_all_weights(self) -> torch.Tensor:
        return torch.cat([param.data.view(-1) for param in self.parameters()])

class AtariNetwork(nn.Module):
    def __init__(self, observation_space: tuple, action_space: int, hidden_layers_sizes: typing.List[int], dropout_p: float):
        super(AtariNetwork, self).__init__()
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(AtariNetwork, self).__init__()
        self.observation_space = observation_space
        self.conv_layers = nn.Sequential(
                        nn.Conv2d(observation_space[2], 32, kernel_size=8, stride=4),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                        nn.ReLU(),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                        nn.ReLU()
        )
        self.fc4 = nn.Linear(self.get_conv_output_size(), 512)
        self.fc5 = nn.Linear(512, action_space)

    def forward(self, x):
        x = x.view(x.size(0), x.size(-1), x.size(1), x.size(2))
        x = self.conv_layers(x)
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)

    def get_conv_output_size(self):
        return self.conv_layers(torch.zeros((1, self.observation_space[2], self.observation_space[0], self.observation_space[1]))).view(1, -1).size(1)

    def get_all_weights(self) -> torch.Tensor:
        return torch.cat([param.data.view(-1) for param in self.parameters()])
        


class DqnAgent(BaseAgent):
    agent_hyper_params = DqnAgentHyperParams()

    def __init__(self, observation_space: tuple, action_space: int, device, debug: bool, current_execution: int, env_name: str):
        super().__init__(observation_space, action_space, device, debug)
        # self.agent_hyper_params = DqnAgentHyperParams(current_execution)
        self.env_name = env_name
        self.exploration_rate = self.agent_hyper_params.exploration_max
        self.memory = ReplayMemory(self.agent_hyper_params.memory_size)

        self.local_model = self.choose_net()(observation_space, action_space, self.agent_hyper_params.hidden_layers_sizes, self.agent_hyper_params.dropout_p).to(self.device, non_blocking=True)
        self.trg_model = self.choose_net()(observation_space, action_space, self.agent_hyper_params.hidden_layers_sizes, self.agent_hyper_params.dropout_p).to(self.device, non_blocking=True)
        self.copy_weigts()
        self.trg_model.eval()

        self.initial_weights: torch.Tensor = self.local_model.get_all_weights()

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

        self.optimizer = optim.Adam(self.local_model.parameters(), lr=self.agent_hyper_params.learning_rate, weight_decay=0)

    def remember(self, state, action, reward, done, next_state):
        super().remember(state, action, reward, done, next_state)
        state = torch.tensor([state], dtype=torch.float).to(self.device, non_blocking=True)
        action = torch.tensor([[action]], dtype=torch.long).to(self.device, non_blocking=True)
        if next_state is not None:
            next_state = torch.tensor([next_state], dtype=torch.float).to(self.device, non_blocking=True)
        reward = torch.tensor([reward], dtype=torch.float).to(self.device, non_blocking=True)
        self.memory.push(state, action, next_state, reward)

    def act(self, state):
        super().act(state)
        if random.random() < self.exploration_rate:
            return random.randrange(self.action_space)
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # if self.agent_hyper_params.frames_to_consider > 1:
            #     state = np.vstack((np.hstack((self.memory.memory[self.memory.position - i].state for i in range(1, self.agent_hyper_params.frames_to_consider))).squeeze(), state))

            return self.local_model(torch.tensor([state], dtype=torch.float).to(self.device, non_blocking=True)).max(1)[1].item()

    def reflect(self, done, step) -> typing.Tuple[typing.Optional[float], typing.Optional[float]]:
        if len(self.memory) < self.agent_hyper_params.batch_size:
            return None, None

        super().reflect(done, step)

        loss = self.__get_loss(self.memory.sample(self.agent_hyper_params.batch_size))
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.exploration_rate *= self.agent_hyper_params.exploration_decay
        self.exploration_rate = max(self.agent_hyper_params.exploration_min, self.exploration_rate)
        # print(self.exploration_rate)

        if step % self.agent_hyper_params.update_trg_model == 0:
            self.copy_weigts()
        return loss.item(), BaseAgent.calc_dist(self.initial_weights, self.local_model.get_all_weights())

    def __get_loss(self, transitions: typing.List[Transition]):
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool).to(self.device, non_blocking=True)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.local_model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the model; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.agent_hyper_params.batch_size).to(self.device, non_blocking=True)

        if self.agent_hyper_params.double_dqn:
            max_action_indices = self.local_model(non_final_next_states).detach().argmax(1)
            next_state_values[non_final_mask] = self.trg_model(non_final_next_states).detach().\
                gather(1, max_action_indices.unsqueeze(1)).squeeze(1)  # double DQN

        else:
            next_state_values[non_final_mask] = self.trg_model(non_final_next_states).detach().max(1)[0]

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.agent_hyper_params.gamma) + reward_batch

        # Compute MSE Loss
        loss = nn.MSELoss()
        # loss = nn.SmoothL1Loss()
        return loss(state_action_values, expected_state_action_values.unsqueeze(1))

    def copy_weigts(self):
        self.trg_model.load_state_dict(self.local_model.state_dict())

    def save_weights(self, path):
        torch.save(self.local_model, os.path.join(path, 'dqn.pth'))

    def set_dropout_p(self, p):
        for l in self.local_model.children():
            if isinstance(l, nn.modules.dropout.Dropout):
                l.p = p

    def get_dropout_p(self):
        for l in self.local_model.children():
            if isinstance(l, nn.modules.dropout.Dropout):
                return l.p

    def choose_net(self):
        if self.env_name == 'CartPole-v1':
            return Network
        elif self.env_name == 'DemonAttack-v0':
            return AtariNetwork
        else:
            return AtariNetwork
        # else:
        #     raise NotImplementedError(f'{self.env_name} is not supported yet.')




