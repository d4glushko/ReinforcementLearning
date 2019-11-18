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
        return state

    def step(self, action):
        state_next, reward, done, info = self.env.step(action)
        return state_next, reward, done, info
