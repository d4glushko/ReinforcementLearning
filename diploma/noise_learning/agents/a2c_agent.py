import random
import typing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

from .base_agent import BaseAgent

# Discount factor. Model is not very sensitive to this value.
GAMMA = .95

# LR of 3e-2 explodes the gradients, LR of 3e-4 trains slower
LR = 3e-3

# OpenAI baselines uses nstep of 5.
N_STEPS = 20


class MemoryCell:
    def __init__(self, state, action, reward: float, done: bool):
        self.state = state
        self.action = action
        self.reward: float = reward
        self.done: bool = done


# https://github.com/rgilman33/simple-A2C/blob/master/3_A2C-nstep-TUTORIAL.ipynb
class A2CAgent(BaseAgent):
    def __init__(self, observation_space: int, action_space: int, debug: bool):
        super().__init__(observation_space, action_space, debug)
        self.gamma: float = GAMMA
        self.n_steps: int = N_STEPS
        self.memory: typing.List[MemoryCell] = []
        self.model = A2CCartPoleNN(observation_space, action_space)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        

    def act(self, state):
        super().act(state)
        s = Variable(torch.from_numpy(state).float().unsqueeze(0))

        action_probs = self.model.get_action_probs(s)

        sample = action_probs.multinomial(self.action_space)
        action_value, action_index = sample.max(dim=1)
        action = action_index.data[0].item()
        return action

    def remember(self, state, action, reward, done, next_state):
        super().remember(state, action, reward, done, next_state)
        self.memory.append(MemoryCell(state, action, reward, done))

    
    def reflect(self):
        if len(self.memory) < self.n_steps:
            return

        super().reflect()
        # Ground trutch labels
        state_values_true = self.__calc_actual_state_values()

        s = Variable(torch.FloatTensor([m.state for m in self.memory]))
        action_probs, state_values_est = self.model.evaluate_actions(s)
        action_log_probs = action_probs.log()

        a = Variable(torch.LongTensor([m.action for m in self.memory]).view(-1,1))
        chosen_action_log_probs = action_log_probs.gather(1, a)

        # TD error
        advantages = state_values_true - state_values_est

        entropy = (action_probs * action_log_probs).sum(1).mean()
        action_gain = (chosen_action_log_probs * advantages).mean()
        value_loss = advantages.pow(2).mean()
        total_loss = value_loss - action_gain - 0.0001 * entropy
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.memory.clear()


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
                this_return = memory_copy[r].reward + next_return * self.gamma

            state_values.append(this_return)
            next_return = this_return

        state_values.reverse()
        result = Variable(torch.FloatTensor(state_values)).unsqueeze(1)

        return result


class A2CCartPoleNN(nn.Module):
    def __init__(self, observation_space: int, action_space: int):
        super(A2CCartPoleNN, self).__init__()
        self.action_space: int = action_space
        self.observation_space: int = observation_space

        self.linear1 = nn.Linear(observation_space, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)

        self.actor = nn.Linear(64, action_space)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        # print(f"Before lin1: {x}")
        # print(f"Lin1 w: {self.linear1.weight}")
        x = self.linear1(x)
        # print(f"Before rel1: {x}")
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
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

    
    