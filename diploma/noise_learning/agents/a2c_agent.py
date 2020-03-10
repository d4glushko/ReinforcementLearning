import random
import typing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.distributions import Categorical

import os

from .base_agent import BaseAgent, AgentHyperParams

# Discount factor. Model is not very sensitive to this value.
GAMMA = .99

# LR of 3e-2 explodes the gradients, LR of 3e-4 trains slower
LR = 1e-4

# OpenAI baselines uses nstep of 5.
N_STEPS = 300

EPS = 1.19209e-07

HIDDEN_LAYERS_SIZES = [256]  # NN hidden layer size
GREEDY_ACTION_SAMPLING = False
DROPOUT_P = 0.
FRAMES_TO_CONSIDER = 1



class A2CAgentHyperParams(AgentHyperParams):

    def __init__(self):
        self.learning_rate: float = LR
        self.gamma: float = GAMMA
        self.steps_number: int = N_STEPS
        self.hidden_layers_sizes: typing.List[int] = HIDDEN_LAYERS_SIZES
        self.eps = EPS
        self.greedy_action_sampling = GREEDY_ACTION_SAMPLING
        self.dropout_p = DROPOUT_P
        self.frames_to_consider: int = FRAMES_TO_CONSIDER



class MemoryCell:
    def __init__(self, state, action, reward: float, done: bool):
        self.state = state
        self.action = action
        self.reward: float = reward
        self.done: bool = done


# https://github.com/rgilman33/simple-A2C/blob/master/3_A2C-nstep-TUTORIAL.ipynb
class A2CAgent(BaseAgent):
    agent_hyper_params = A2CAgentHyperParams()

    def __init__(self, observation_space: tuple, action_space: int, device, debug: bool, current_execution: int, env_name: str):
        super().__init__(observation_space, action_space, device, debug)

        self.memory: typing.List[MemoryCell] = []
        self.model = A2CCartPoleNN(observation_space, action_space, self.agent_hyper_params.hidden_layers_sizes,
                                   self.agent_hyper_params.dropout_p).to(self.device)
        self.initial_weights: torch.Tensor = self.model.get_all_weights()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.agent_hyper_params.learning_rate)
        

    def act(self, state):
        super().act(state)
        s = Variable(torch.from_numpy(state).float().unsqueeze(0))

        action_probs = self.model.get_action_probs(s)
        if self.agent_hyper_params.greedy_action_sampling:
            action = action_probs.argmax()
        else:
            m = Categorical(action_probs)
            action = m.sample()

        # sample = action_probs.multinomial(self.action_space)
        # action_value, action_index = sample.max(dim=1)
        # action = action_index.data[0].item()
        return action.item()

    def remember(self, state, action, reward, done, next_state):
        super().remember(state, action, reward, done, next_state)
        self.memory.append(MemoryCell(state, action, reward, done))

    
    def reflect(self, done, step) -> typing.Tuple[typing.Optional[float], typing.Optional[float]]:
        # if len(self.memory) < self.agent_hyper_params.steps_number:
        #     return None, None

        if not done:
            return None, None

        super().reflect(done, step)
        # Ground truth labels
        state_values_true = self.__calc_actual_state_values()

        s = Variable(torch.FloatTensor([m.state for m in self.memory], device=self.device))
        action_probs, state_values_est = self.model.evaluate_actions(s)
        action_log_probs = action_probs.log()

        a = Variable(torch.LongTensor([m.action for m in self.memory], device=self.device).view(-1,1))
        chosen_action_log_probs = action_log_probs.gather(1, a)

        # TD error
        advantages = state_values_true - state_values_est

        entropy = (action_probs * action_log_probs).sum(1).mean()
        action_gain = (chosen_action_log_probs * advantages).sum()
        # value_loss = advantages.pow(2).mean()
        value_loss = F.smooth_l1_loss(state_values_est, state_values_true, reduction='sum')
        total_loss = value_loss - action_gain - 0.0001 * entropy
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.memory.clear()
        return total_loss.item(), BaseAgent.calc_dist(self.initial_weights, self.model.get_all_weights())


    def __calc_actual_state_values(self):
        memory_copy = self.memory.copy()

        next_return: float = 0
        if not memory_copy[-1].done:
            s = torch.from_numpy(memory_copy[-1].state).float().unsqueeze(0)
            next_return = self.model.get_state_value(Variable(s)).data[0][0]

        state_values: typing.List[float] = []
        state_values.append(next_return)
        memory_copy.reverse()
        
        for r in range(1, len(memory_copy)):
            this_return: float = 0
            if not memory_copy[r].done:
                this_return = memory_copy[r].reward + next_return * self.agent_hyper_params.gamma

            state_values.append(this_return)
            next_return = this_return

        state_values.reverse()

        result = torch.FloatTensor(state_values, device=self.device)
        # result = (result - result.mean()) / (result.std() + self.agent_hyper_params.eps)
        result = Variable(result).unsqueeze(1)

        return result

    def save_weights(self, path):
        torch.save(self.model, os.path.join(path, 'a2c.pth'))

    def set_dropout_p(self, p):
        for l in self.model.children():
            if isinstance(l, nn.modules.dropout.Dropout):
                l.p = p

    def get_dropout_p(self):
        for l in self.model.children():
            if isinstance(l, nn.modules.dropout.Dropout):
                return l.p


class A2CCartPoleNN(nn.Module):
    def __init__(self, observation_space: tuple, action_space: int, hidden_layers_sizes: typing.List[int], dropout_p: float):
        super(A2CCartPoleNN, self).__init__()
        self.action_space: int = action_space
        self.observation_space: tuple = observation_space

        self.linear1 = nn.Linear(observation_space[0], hidden_layers_sizes[0])
        # self.linear2 = nn.Linear(hidden_layers_sizes[0], hidden_layers_sizes[1])
        # self.linear3 = nn.Linear(hidden_layers_sizes[1], hidden_layers_sizes[2])
        self.dr1 = nn.Dropout(p=dropout_p)
        # self.dr2 = nn.Dropout(p=dropout_p)

        self.actor = nn.Linear(hidden_layers_sizes[0], action_space)
        self.critic = nn.Linear(hidden_layers_sizes[0], 1)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dr1(x)
        # x = self.linear2(x)
        # x = F.relu(x)
        # x = self.dr2(x)
        return x

    # Actor head
    def get_action_probs(self, x):
        x = self(x)
        x = self.__get_action_probs_part(x)
        return x

    # Critic head
    def get_state_value(self, x):
        x = self(x)
        x = self.__get_state_value_part(x)
        return x

    def evaluate_actions(self, x):
        x = self(x)
        action_probs = self.__get_action_probs_part(x)
        state_values = self.__get_state_value_part(x)
        return action_probs, state_values

    def __get_action_probs_part(self, x):
        x = self.actor(x)
        x = F.softmax(x, dim=1)
        return x

    def __get_state_value_part(self, x):
        x = self.critic(x)
        return x

    def get_all_weights(self) -> torch.Tensor:
        return torch.cat([param.data.view(-1) for param in self.parameters()])
    