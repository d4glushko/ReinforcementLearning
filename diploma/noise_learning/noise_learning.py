import typing
import numpy as np
from enum import Enum

from .agents.base_agent import BaseAgent
from .agents.a2c_agent import A2CAgent
from .agents.dqn_agent import DqnAgent
from .envs.env import EnvironmentWrapper, NoiseType


class NoiseLearningAgents(Enum):
    DQN = 1
    A2C = 2
    


class NoiseLearning:
    def __init__(self, agents_number: int, env_name: str, noise_learning_agent: NoiseLearningAgents):
        self.agents_number: int = agents_number
        self.environments: typing.List[EnvironmentWrapper] = [
            EnvironmentWrapper(env_name, NoiseType.Random) for i in range(agents_number)
        ]
        self.agents: typing.List[BaseAgent] = [
            self.__choose_agent(noise_learning_agent)(env.observation_space(), env.action_space())
            for env in [
                self.environments[i]
                for i in range(agents_number)
            ]
        ]

    def __choose_agent(self, noise_learning_agent: NoiseLearningAgents):
        agents_mapping = {
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
                    reward = reward if not done else -reward
                    agent.remember(state, action, reward, done, state_next)
                    state = state_next
                    if done:
                        break

                    experience_replay = getattr(agent, "experience_replay", None)
                    if callable(experience_replay):
                        agent.experience_replay()

                print(f"Agent {j} finished. Score {score}")

            if self.should_swap_agents():
                self.swap_agents()

    def should_swap_agents(self):
        pass

    def swap_agents(self):
        pass
