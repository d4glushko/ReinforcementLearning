import typing

from enum import Enum
import gym

from .agents.base_agent import BaseAgent
from .agents.a2c_agent import A2CAgent
from .agents.dqn_agent import DqnAgent

from .envs.env import FrameStack, FrameProcess, AddAtariNoise, AddCartPoleNoise

class NoiseLearningAgents(Enum):
    DQN = 1
    A2C = 2


class ExchangeTypes(Enum):
    NO = 1
    RANDOM = 2
    SMART = 3


def choose_agent(noise_learning_agent: NoiseLearningAgents) -> typing.Type[BaseAgent]:
    agents_mapping = {
        NoiseLearningAgents.DQN: DqnAgent,
        NoiseLearningAgents.A2C: A2CAgent
    }
    return agents_mapping[noise_learning_agent]


def choose_environment_wrapper(env_n: str, num_frames: int, noise_std_dev):
    env = gym.make(env_n)
    env_mapping = {'CartPole-v1': AddCartPoleNoise(env, noise_std_dev),
                   'Breakout-v4': FrameStack(AddAtariNoise(FrameProcess(env), noise_std_dev), num_frames)}
    return env_mapping[env_n]