import typing
import numpy as np
from enum import Enum

from .agents.base_agent import BaseAgent
from .agents.a2c_agent import A2CAgent
from .agents.dqn_agent import DqnAgent
from .agents.test_agent import TestAgent
from .envs.env import EnvironmentWrapper, NoiseType


class NoiseLearningAgents(Enum):
    TEST = 1
    DQN = 2
    A2C = 3
    

class NoiseLearning:
    def __init__(self, agents_number: int, env_name: str, noise_learning_agent: NoiseLearningAgents, debug: bool = False):
        self.agents_number: int = agents_number
        self.environments: typing.List[EnvironmentWrapper] = [
            EnvironmentWrapper(env_name, NoiseType.Random) for i in range(agents_number)
        ]
        self.agents: typing.List[BaseAgent] = [
            self.__choose_agent(noise_learning_agent)(env.observation_space(), env.action_space(), debug)
            for env in [
                self.environments[i]
                for i in range(agents_number)
            ]
        ]

    def __choose_agent(self, noise_learning_agent: NoiseLearningAgents) -> typing.Type[BaseAgent]:
        agents_mapping = {
            NoiseLearningAgents.TEST: TestAgent,
            NoiseLearningAgents.DQN: DqnAgent,
            NoiseLearningAgents.A2C: A2CAgent
        }
        return agents_mapping[noise_learning_agent]

    def train(self, training_episodes: int = 1000):
        for i in range(training_episodes):
            print(f"Episode {i}")
            for j in range(self.agents_number):
                print(f"Agent {j} started")
                agent = self.agents[j]
                env = self.environments[j]

                state = env.reset()

                # TODO: code is bound to the CartPole env currently. Make it more env agnostic
                score = 0
                while True:
                    env.render()
                    score += 1
                    action = agent.act(state)
                    
                    state_next, reward, done, info = env.step(action)
                    if done:
                        reward = -reward
                        state_next = None
                    
                    agent.remember(state, action, reward, done, state_next)
                    agent.reflect()

                    if done:
                        break

                    state = state_next

                print(f"Agent {j} finished. Score {score}")

            if self.should_swap_agents():
                self.swap_agents()

    def should_swap_agents(self):
        pass

    def swap_agents(self):
        pass
