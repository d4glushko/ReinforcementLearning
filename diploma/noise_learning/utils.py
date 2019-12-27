import typing

from enum import Enum

from .agents.base_agent import BaseAgent
from .agents.a2c_agent import A2CAgent
from .agents.dqn_agent import DqnAgent


class NoiseLearningAgents(Enum):
    DQN = 1
    A2C = 2

   
def choose_agent(noise_learning_agent: NoiseLearningAgents) -> typing.Type[BaseAgent]:
    agents_mapping = {
        NoiseLearningAgents.DQN: DqnAgent,
        NoiseLearningAgents.A2C: A2CAgent
    }
    return agents_mapping[noise_learning_agent]