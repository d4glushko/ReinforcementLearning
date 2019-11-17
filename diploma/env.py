from enum import Enum

import gym
import numpy as np


class NoiseType(Enum):
    Empty = 1,
    Random = 2


class EnvironmentWrapper:
    def __init__(self, env_name: str, noise_type: NoiseType):
        self.env = gym.make(env_name)

    def observation_space(self):
        return self.env.observation_space.shape[0]

    def action_space(self):
        return self.env.action_space.n

    def reset(self):
        state = self.env.reset()
        return np.reshape(state, [1, self.observation_space()])

    def step(self, action):
        state_next, reward, terminal, info = self.env.step(action)
        reward = reward if not terminal else -reward
        state_next = np.reshape(state_next, [1, self.observation_space()])
        return state_next, reward, terminal, info
