import typing
import numpy as np
import random
import matplotlib
import torch
import math
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from enum import Enum

from .utils import NoiseLearningAgents, ExchangeTypes, choose_agent
from .agents.base_agent import BaseAgent
from .agents.a2c_agent import A2CAgent
from .agents.dqn_agent import DqnAgent
from .envs.env import EnvironmentWrapper
from .results_manager import ResultsManager, Settings, AgentResults
    

class NoiseLearning:
    def __init__(
        self, exchange_type: ExchangeTypes, exchange_delta: float, exchange_items_reward_count: int, training_episodes: int, agents_number: int, 
        env_name: str, noise_learning_agent: NoiseLearningAgents, debug: bool, noise_env_step: float, epsilon_wrt_noise: bool, use_cuda: bool, warm_up_steps: int,
        exchange_steps: int, date: int, current_execution: int = 1, total_executions: int = 1,
    ):
        if exchange_type != ExchangeTypes.NO and agents_number < 2:
            raise Exception(f"Agents number must be >= 2 for {exchange_type.name} exchange_type. Current value: {agents_number}")

        self.exchange_delta: float = exchange_delta
        self.exchange_items_reward_count: int = exchange_items_reward_count
        self.warm_up_steps: int = warm_up_steps
        self.exchange_steps: int = exchange_steps
        self.exchange_type: ExchangeTypes = exchange_type
        self.training_episodes: int = training_episodes
        self.agents_number: int = agents_number
        self.noise_learning_agent: NoiseLearningAgents = noise_learning_agent
        self.noise_env_step: float = noise_env_step
        self.epsilon_wrt_noise: bool = epsilon_wrt_noise
        self.env_name: str = env_name
        self.use_cuda: bool = use_cuda
        self.debug: bool = debug

        self.date: int = date
        self.current_execution: int = current_execution
        self.total_executions: int = total_executions

        self.__setup_envs()
        self.__setup_agents()
        self.__setup_agents_results()

    def train(self):
        running_scores = [10] * self.agents_number

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

                self.__train_agent_episode(agent, env, agent_results, i, j, running_scores)

            self.__perform_exchange(i)

    def save_results(self):
        self.results_manager.save_results(self.agents_results, self.date, self.current_execution)

    def __increase_agents_exchange_attempts(self):
        for agent_result in self.agents_results:
            agent_result.exchange_attempts = agent_result.exchange_attempts + 1

    def __perform_exchange(self, iter: int):
        if not (iter % self.exchange_steps == 0 and iter >= self.warm_up_steps):
            return

        if self.exchange_type == ExchangeTypes.NO:
            return
        elif self.exchange_type == ExchangeTypes.RANDOM:
            self.__increase_agents_exchange_attempts()
            self.__perform_random_exchange()
        elif self.exchange_type == ExchangeTypes.SMART:
            self.__increase_agents_exchange_attempts()
            self.__perform_smart_exchange(iter)
        return

    def __perform_random_exchange(self):
        if not self.__should_random_exchange():
            return

        agent_number = random.randrange(self.agents_number)
        if agent_number == 0:
            self.__swap_environments(agent_number, agent_number + 1)
        elif agent_number == self.agents_number - 1:
            self.__swap_environments(agent_number - 1, agent_number)
        elif random.random() < 0.5:
            self.__swap_environments(agent_number, agent_number + 1)
        else:
            self.__swap_environments(agent_number - 1, agent_number)

    def __should_random_exchange(self):
        # Idea is to swap each agent once per every 100 iterations (for CartPole DQN) on average. 
        iterations_count = 100 / self.exchange_steps
        chance = (1 / iterations_count) * (self.agents_number / 2) # because 2 agents are participating in swap
        return random.random() < chance

    def __perform_smart_exchange(self, iter: int):
        direction = int(iter / self.exchange_steps) % 2 == 0
        if direction:
            for i in range(self.agents_number - 1):
                self.__smart_exchange_agents(i, i + 1)
        else:
            for i in range(self.agents_number - 1, 0, -1):
                self.__smart_exchange_agents(i - 1, i)

    def __smart_exchange_agents(self, idx: int, next_idx: int):
        noise = self.environments[idx].noise_std_dev
        next_noise = self.environments[next_idx].noise_std_dev

        cumulative_reward = np.array(
            [
                metric.value for metric in self.agents_results[idx].scores.metrics
            ][-self.exchange_items_reward_count:]
        ).mean()
        next_cumulative_reward = np.array(
            [
                metric.value for metric in self.agents_results[next_idx].scores.metrics
            ][-self.exchange_items_reward_count:]
        ).mean()

        formula = math.exp(
            self.exchange_delta * (next_noise - noise) * (next_cumulative_reward - cumulative_reward)
        )
        chance = min(formula, 1)
        if random.random() < chance:
            self.__swap_environments(idx, next_idx)

    def __swap_environments(self, idx1, idx2):
        env_buf = self.environments[idx1]
        self.environments[idx1] = self.environments[idx2]
        self.environments[idx2] = env_buf

        self.agents_results[idx1].exchanges = self.agents_results[idx1].exchanges + 1
        self.agents_results[idx2].exchanges = self.agents_results[idx2].exchanges + 1

    def __setup_envs(self):
        self.environments: typing.List[EnvironmentWrapper] = [
            EnvironmentWrapper(self.env_name, noise_std_dev=(i * self.noise_env_step)) for i in range(self.agents_number)
        ]

    def __setup_agents(self):
        self.agents: typing.List[BaseAgent] = [
            choose_agent(self.noise_learning_agent)(
                env.observation_space(), env.action_space(), self.__select_device(i, self.use_cuda), self.debug
            )
            for env, i in [
                (self.environments[i], i)
                for i in range(self.agents_number)
            ]
        ]
        if self.epsilon_wrt_noise:
            for i, agent in enumerate(self.agents):
                if hasattr(agent, 'exploration_rate'):
                    agent.exploration_rate = 1 - i * self.noise_env_step

    def __setup_agents_results(self):
        agent_hyper_params = choose_agent(self.noise_learning_agent).agent_hyper_params.to_dict()
        self.results_manager: ResultsManager = ResultsManager(
            Settings(
                self.agents_number, self.env_name, self.noise_learning_agent.name, self.noise_env_step, self.exchange_type.name, 
                self.exchange_delta, self.exchange_items_reward_count, agent_hyper_params
            )
        )
        self.agents_results: typing.List[AgentResults] = [
            AgentResults() for i in range(self.agents_number)
        ]

    def __train_agent_episode(
        self, agent: BaseAgent, env: EnvironmentWrapper, agent_results: AgentResults, iteration: int, agent_number: int, running_scores):
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
            loss, dist = agent.reflect(done)
            agent_results.add_loss(loss, iteration, env.noise_std_dev)
            agent_results.add_dist(dist, iteration, env.noise_std_dev)

            if done:
                break

            state = state_next
        agent_results.add_score(score, iteration, env.noise_std_dev)
        running_scores[agent_number] = 0.05 * score + (1 - 0.05) * running_scores[agent_number]

        print(f"Agent {agent_number} finished. Score {score}. Running score {running_scores[agent_number]}")

    def __select_device(self, agent_number, use_cuda):
        cuda_available = torch.cuda.is_available()
        device = None
        if use_cuda and cuda_available:
            cuda_count = torch.cuda.device_count()
            device = torch.device(f"cuda:{agent_number % cuda_count}")
        else:
            device = torch.device("cpu")
        return device
