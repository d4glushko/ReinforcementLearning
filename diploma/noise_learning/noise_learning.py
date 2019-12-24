import typing
import numpy as np
import random
import matplotlib
import torch
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from enum import Enum

from .agents.base_agent import BaseAgent
from .agents.a2c_agent import A2CAgent
from .agents.dqn_agent import DqnAgent
from .envs.env import EnvironmentWrapper
from .results_manager import ResultsManager, Settings, AgentResults


class NoiseLearningAgents(Enum):
    DQN = 1
    A2C = 2
    

class NoiseLearning:
    def __init__(
        self, enable_exchange: bool, training_episodes: int, agents_number: int, env_name: str, noise_learning_agent: NoiseLearningAgents, 
        debug: bool, noise_env_step: float, use_cuda: bool, current_execution: int = 1, total_executions: int = 1
    ):
        self.enable_exchange: bool = enable_exchange
        self.training_episodes: int = training_episodes
        self.agents_number: int = agents_number
        self.noise_learning_agent: NoiseLearningAgents = noise_learning_agent
        self.noise_env_step: float = noise_env_step
        self.env_name: str = env_name
        self.use_cuda: bool = use_cuda
        self.debug: bool = debug

        self.current_execution: int = current_execution
        self.total_executions: int = total_executions

        self.__setup_envs()
        self.__setup_agents()
        self.__setup_agents_results()

    def train(self):
        for i in range(1, self.training_episodes + 1):
            current_execution_percent = i / self.training_episodes * 100
            total_percent = (current_execution_percent + 100 * (self.current_execution - 1)) / self.total_executions
            print(
                f"Episode {i}. Execution {current_execution_percent:.2f}% done. "
                f"{self.current_execution}/{self.total_executions} execution. Total {total_percent:.2f}% done."
            )
            for j in range(self.agents_number):
                agent = self.agents[j]
                env = self.environments[j]
                agent_results = self.agents_results[j]

                self.__train_agent_episode(agent, env, agent_results, i, j)

            self.__perform_random_swap()

    def save_results(self):
        self.results_manager.save_results(self.agents_results)

    def __perform_random_swap(self):
        if not self.__should_swap_agents():
            return

        agent_number = random.randrange(self.agents_number)
        if agent_number == 0:
            self.__swap_environments(agent_number, agent_number + 1)
        elif agent_number == self.agents_number:
            self.__swap_environments(agent_number - 1, agent_number)
        elif random.random() < 0.5:
            self.__swap_environments(agent_number, agent_number + 1)
        else:
            self.__swap_environments(agent_number - 1, agent_number)

    def __should_swap_agents(self):
        if not self.enable_exchange or self.agents_number == 1:
            return False

        # Idea is to swap each agent once per every 100 iterations (for CartPole DQN) on average
        iterations_count = 100
        chance = (1 / iterations_count) * (self.agents_number / 2) # because 2 agents are participating in swap
        return random.random() < chance

    def __swap_environments(self, idx1, idx2):
        env_buf = self.environments[idx1]
        self.environments[idx1] = self.environments[idx2]
        self.environments[idx2] = env_buf

    def __setup_envs(self):
        self.environments: typing.List[EnvironmentWrapper] = [
            EnvironmentWrapper(self.env_name, noise_std_dev=(i * self.noise_env_step)) for i in range(self.agents_number)
        ]

    def __setup_agents(self):
        self.agents: typing.List[BaseAgent] = [
            self.__choose_agent(self.noise_learning_agent)(
                env.observation_space(), env.action_space(), self.__select_device(i, self.use_cuda), self.debug
            )
            for env, i in [
                (self.environments[i], i)
                for i in range(self.agents_number)
            ]
        ]

    def __setup_agents_results(self):
        self.results_manager: ResultsManager = ResultsManager(
            Settings(self.agents_number, self.env_name, self.noise_learning_agent.name, self.noise_env_step)
        )
        self.agents_results: typing.List[AgentResults] = [
            AgentResults() for i in range(self.agents_number)
        ]

    def __choose_agent(self, noise_learning_agent: NoiseLearningAgents) -> typing.Type[BaseAgent]:
        agents_mapping = {
            NoiseLearningAgents.DQN: DqnAgent,
            NoiseLearningAgents.A2C: A2CAgent
        }
        return agents_mapping[noise_learning_agent]

    def __train_agent_episode(
        self, agent: BaseAgent, env: EnvironmentWrapper, agent_results: AgentResults, iteration: int, agent_number: int
    ):
        state = env.reset()

        # TODO: code is bound to the CartPole env currently. Make it more env agnostic
        score = 0
        while True:
            # env.render()
            score += 1

            action = agent.act(state)
            state_next, reward, done, info = env.step(action)
            
            if done:
                reward = -reward
                state_next = None
            
            agent.remember(state, action, reward, done, state_next)
            agent.reflect()
            agent_results.add_loss(agent.last_loss, iteration, env.noise_std_dev)

            if done:
                break

            state = state_next
        agent_results.add_score(score, iteration, env.noise_std_dev)
        print(f"Agent {agent_number} finished. Score {score}")

    def __select_device(self, agent_number, use_cuda):
        cuda_available = torch.cuda.is_available()
        device = None
        if use_cuda and cuda_available:
            cuda_count = torch.cuda.device_count()
            device = torch.device(f"cuda:{agent_number % cuda_count}")
        else:
            device = torch.device("cpu")
        return device
