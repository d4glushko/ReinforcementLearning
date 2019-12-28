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

from .base_agent import BaseAgent, AgentHyperParams


# hyper parameters
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
GAMMA = 0.9  # Q-learning discount factor
LR = 0.001  # NN optimizer learning rate
HIDDEN_LAYERS_SIZES = [24, 24]  # NN hidden layer size
BATCH_SIZE = 128  # Q-learning batch size

MEMORY_SIZE = 300000

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DqnAgentHyperParams(AgentHyperParams):
    def __init__(self):
        self.learning_rate: float = LR
        self.gamma: float = GAMMA
        self.hidden_layers_sizes: typing.List[int] = HIDDEN_LAYERS_SIZES
        self.batch_size: int = BATCH_SIZE
        self.memory_size: int = MEMORY_SIZE
        self.exploration_max: float = EXPLORATION_MAX
        self.exploration_min: float = EXPLORATION_MIN
        self.exploration_decay: float = EXPLORATION_DECAY


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
    def __init__(self, observation_space: int, action_space: int, hidden_layers_sizes: typing.List[int]):
        super(Network, self).__init__()
        self.l1 = nn.Linear(observation_space, hidden_layers_sizes[0])
        self.l2 = nn.Linear(hidden_layers_sizes[0], hidden_layers_sizes[1])
        self.l3 = nn.Linear(hidden_layers_sizes[1], action_space)

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
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

    def get_all_weights(self) -> torch.Tensor:
        return torch.cat([param.data.view(-1) for param in self.parameters()])

class DqnAgent(BaseAgent):
    agent_hyper_params = DqnAgentHyperParams()

    def __init__(self, observation_space: int, action_space: int, device, debug: bool):
        super().__init__(observation_space, action_space, device, debug)
        self.exploration_rate = self.agent_hyper_params.exploration_max
        self.memory = ReplayMemory(self.agent_hyper_params.memory_size)

        self.model = Network(observation_space, action_space, self.agent_hyper_params.hidden_layers_sizes).to(self.device, non_blocking=True)
        self.initial_weights: torch.Tensor = self.model.get_all_weights()

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

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.agent_hyper_params.learning_rate)

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
            return self.model(torch.tensor([state], dtype=torch.float).to(self.device, non_blocking=True)).max(1)[1].item()

    def reflect(self, done) -> typing.Tuple[typing.Optional[float], typing.Optional[float]]:
        if len(self.memory) < self.agent_hyper_params.batch_size:
            return None, None

        super().reflect(done)

        loss = self.__get_loss(self.memory.sample(self.agent_hyper_params.batch_size))
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.exploration_rate *= self.agent_hyper_params.exploration_decay
        self.exploration_rate = max(self.agent_hyper_params.exploration_min, self.exploration_rate)
        return loss.item(), BaseAgent.calc_dist(self.initial_weights, self.model.get_all_weights())

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
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the model; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.agent_hyper_params.batch_size).to(self.device, non_blocking=True)
        next_state_values[non_final_mask] = self.model(non_final_next_states).detach().max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.agent_hyper_params.gamma) + reward_batch

        # Compute MSE Loss
        loss = nn.MSELoss()
        return loss(state_action_values, expected_state_action_values.unsqueeze(1))
