import gym
import numpy as np
from enum import Enum


class EnvironmentWrapper:
    def __init__(self, env_name: str, noise_std_dev: float = 0):
        self.env = gym.make(env_name)
        self.noise_std_dev: float = noise_std_dev
        self.is_noise: bool = not (self.noise_std_dev == 0)

    def observation_space(self):
        return self.env.observation_space.shape[0]

    def action_space(self):
        return self.env.action_space.n

    def reset(self):
        state = self.env.reset()
        state = self.__noised_state(state)
        return state

    def render(self):
        return self.env.render()

    def step(self, action):
        state_next, reward, done, info = self.env.step(action)
        state_next = self.__noised_state(state_next)
        return state_next, reward, done, info

    def __noised_state(self, state):
        return state * self.__sample_scale_noise()

    def __sample_scale_noise(self) -> float:
        mean = 1
        noise = mean
        if self.is_noise:
            noise = np.random.normal(mean, self.noise_std_dev)
        return noise
