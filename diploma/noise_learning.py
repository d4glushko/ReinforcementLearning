import typing

import numpy as np

from agent import Agent
from env import EnvironmentWrapper, NoiseType


class NoiseLearning:
    def __init__(self, agents_number: int, env_name: str):
        self.agents_number: int = agents_number
        self.environments: typing.List[EnvironmentWrapper] = [
            EnvironmentWrapper(env_name, NoiseType.Random) for i in range(agents_number)
        ]
        self.agents: typing.List[Agent] = [
            Agent(env.observation_space(), env.action_space())
            for env in [
                self.environments[i]
                for i in range(agents_number)
            ]
        ]

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
                    score += 1
                    action = agent.act(state)
                    
                    state_next, reward, done, info = env.step(action)
                    agent.remember(state, action, reward, done)
                    state = state_next
                    if done:
                        break
                print(f"Agent {j} finished. Score {score}")

            if self.should_swap_agents():
                self.swap_agents()

    def should_swap_agents(self):
        pass

    def swap_agents(self):
        pass
