import typing
import numpy as np
import random
import matplotlib
import torch
import math
import time
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(color_codes=True)
from matplotlib.cm import get_cmap
from enum import Enum
import gym

from .utils import NoiseLearningAgents, ExchangeTypes, choose_agent, choose_environment_wrapper
from .agents.base_agent import BaseAgent
from .agents.a2c_agent import A2CAgent
from .agents.dqn_agent import DqnAgent

from .results_manager import ResultsManager, Settings, AgentResults
    

class NoiseLearning:
    def __init__(
        self, exchange_type: ExchangeTypes, exchange_delta: float, exchange_items_reward_count: int, training_episodes: int, num_steps_per_episode: int, play_episodes: int, agents_number: int,
        env_name: str, noise_learning_agent: NoiseLearningAgents, debug: bool, noise_env_step: float, noise_dropout_step: float, epsilon_wrt_noise: bool, use_cuda: bool, warm_up_steps: int,
        exchange_steps: int, early_stopping: bool, date: int, current_execution: int = 1, total_executions: int = 1,
    ):
        if exchange_type != ExchangeTypes.NO and agents_number < 2:
            raise Exception(f"Agents number must be >= 2 for {exchange_type.name} exchange_type. Current value: {agents_number}")

        self.exchange_delta: float = exchange_delta
        self.exchange_items_reward_count: int = exchange_items_reward_count
        self.warm_up_steps: int = warm_up_steps
        self.exchange_steps: int = exchange_steps
        self.early_stopping = early_stopping
        self.exchange_type: ExchangeTypes = exchange_type
        self.training_episodes: int = training_episodes
        self.num_steps_per_episode = num_steps_per_episode
        self.play_episodes: int = play_episodes
        self.agents_number: int = agents_number
        self.noise_learning_agent: NoiseLearningAgents = noise_learning_agent
        self.noise_env_step: float = noise_env_step
        self.noise_dropout_step: float = noise_dropout_step
        self.epsilon_wrt_noise: bool = epsilon_wrt_noise
        self.env_name: str = env_name
        self.use_cuda: bool = use_cuda
        self.debug: bool = debug

        self.date: int = date
        self.current_execution: int = current_execution
        self.total_executions: int = total_executions

        self.init_states = []

        self.__setup_envs()
        self.__setup_agents()
        self.__setup_agents_results()


    def train(self):
        running_scores = [10] * self.agents_number
        num_steps = [0] * self.agents_number

        for i in range(1, self.training_episodes + 1):
            current_execution_percent = i / self.training_episodes * 100
            total_percent = (current_execution_percent + 100 * (self.current_execution - 1)) / self.total_executions
            print(
                f"Train Episode {i}. Execution {current_execution_percent:.2f}% done. "
                f"{self.current_execution}/{self.total_executions} execution. Total {total_percent:.2f}% done."
            )
            for j in range(self.agents_number):
                agent = self.agents[j]
                env = self.environments[j]
                agent_results = self.agents_results[j]
                if self.num_steps_per_episode:
                    self.train_agent_steps(agent, env, agent_results, i, j, running_scores, num_steps)
                else:
                    self.__train_agent_episode(agent, env, agent_results, i, j, running_scores, num_steps)

            self.__perform_exchange(i)

            # if i % 500 == 0:
            #     plt.clf()
            #     for j in range(4):
            #         sns.distplot(np.array(self.init_states)[:, j])
            #     plt.savefig(f'obs_e_{i}.png')
            #     self.init_states = []

            if self.early_stopping and np.any(np.array(running_scores) > 498):
                print('Stopping early')
                break


        print(num_steps)


    def play(self):
        env = next((e for e in self.environments if not e.is_noise), None)
        if env == None:
            raise Exception("Environment without noise was not found!")
        for i in range(1, self.play_episodes + 1):
            current_execution_percent = i / self.play_episodes * 100
            print(
                f"Play Episode {i}. Execution {current_execution_percent:.2f}% done. "
            )
            for j in range(self.agents_number):
                agent = self.agents[j]
                agent_play_results = self.agents_play_results[j]

                self.__play_agent_episode(agent, env, agent_play_results, i, j)

    def save_train_results(self):
        self.results_manager.save_train_results(self.agents_results, self.date, self.current_execution)
        for a in self.agents:
            a.save_weights(self.results_manager.target_path)

    def save_play_results(self):
        self.results_manager.save_play_results(self.agents_play_results, self.date, self.current_execution)

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
        if self.noise_dropout_step:
            raise NotImplementedError('Rnd exchange for dropout is not implemeted yet')
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
        if self.noise_env_step:
            noise = self.environments[idx].noise_std_dev
            next_noise = self.environments[next_idx].noise_std_dev
        elif self.noise_dropout_step:
            noise = self.agents[idx].get_dropout_p()
            next_noise = self.agents[next_idx].get_dropout_p()
        else:
            return

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
            if self.noise_env_step:
                self.__swap_environments(idx, next_idx)
            elif self.noise_dropout_step:
                self.__swap_dropouts(idx, next_idx)

    def __swap_environments(self, idx1, idx2):
        env_buf = self.environments[idx1]
        self.environments[idx1] = self.environments[idx2]
        self.environments[idx2] = env_buf

        self.agents_results[idx1].exchanges = self.agents_results[idx1].exchanges + 1
        self.agents_results[idx2].exchanges = self.agents_results[idx2].exchanges + 1

    def __swap_dropouts(self, idx1, idx2):
        d_buf = self.agents[idx1].get_dropout_p()
        self.agents[idx1].set_dropout_p(self.agents[idx2].get_dropout_p())
        self.agents[idx2].set_dropout_p(d_buf)

        self.agents_results[idx1].exchanges = self.agents_results[idx1].exchanges + 1
        self.agents_results[idx2].exchanges = self.agents_results[idx2].exchanges + 1

    def __setup_envs(self):
        self.environments: typing.List[gym.Env] = [
            choose_environment_wrapper(self.env_name, num_frames=4, noise_std_dev=(i * self.noise_env_step)) for i in range(self.agents_number)
        ]

    def __setup_agents(self):
        self.agents: typing.List[BaseAgent] = [
            choose_agent(self.noise_learning_agent)(
                env.observation_space.shape, env.action_space.n, self.__select_device(i, self.use_cuda), self.debug, self.current_execution, self.env_name
            )
            for env, i in [
                (self.environments[i], i)
                for i in range(self.agents_number)
            ]
        ]
        if self.noise_dropout_step and len(self.agents) > 1:
            for i, agent in enumerate(self.agents):
                if hasattr(agent, 'set_dropout_p'):
                    agent.set_dropout_p(i * self.noise_dropout_step)
        if self.epsilon_wrt_noise:
            for i, agent in enumerate(self.agents):
                if hasattr(agent, 'exploration_rate'):
                    agent.exploration_rate = 1 - i * self.noise_env_step

    def __setup_agents_results(self):
        # agent_hyper_params = choose_agent(self.noise_learning_agent).agent_hyper_params.to_dict()
        agent_hyper_params = self.agents[0].agent_hyper_params.to_dict()
        self.results_manager: ResultsManager = ResultsManager(
            Settings(
                self.agents_number, self.env_name, self.noise_learning_agent.name, self.noise_env_step, self.noise_dropout_step,
                self.early_stopping, self.exchange_type.name,
                self.exchange_delta, self.exchange_items_reward_count, self.num_steps_per_episode, agent_hyper_params
            )
        )
        self.agents_results: typing.List[AgentResults] = [
            AgentResults() for i in range(self.agents_number)
        ]
        
        self.agents_play_results: typing.List[AgentResults] = [
            AgentResults() for i in range(self.agents_number)
        ]

    def __play_agent_episode(
        self, agent: BaseAgent, env: gym.Env, agent_play_results: AgentResults, iteration: int, agent_number: int):
        state = env.reset()
        score = 0
        while True:
            # env.render()
            score += 1

            action = agent.act(state)
            state_next, reward, done, info = env.step(action)

            if done:
                break

            state = state_next
        agent_play_results.add_score(score, iteration, env.noise_std_dev if env.noise_std_dev else agent.get_dropout_p())

        print(f"Agent {agent_number} finished. Score {score}")

    def __train_agent_episode(
        self, agent: BaseAgent, env: gym.Env, agent_results: AgentResults, iteration: int, agent_number: int,
            running_scores, num_steps):

        state = env.reset()
        self.init_states.append(state)

        score = 0
        while True:
            env.render()

            num_steps[agent_number] += 1

            action = agent.act(state)
            state_next, reward, done, info = env.step(action)
            score += 1
            
            if done:
                state_next = None
            
            agent.remember(state, action, reward, done, state_next)
            loss, dist = agent.reflect(done, num_steps[agent_number])
            agent_results.add_loss(loss, iteration, env.noise_std if env.noise_std else agent.get_dropout_p())
            agent_results.add_dist(dist, iteration, env.noise_std if env.noise_std else agent.get_dropout_p())

            if done:
                break

            state = state_next

        agent_results.add_score(score, iteration, env.noise_std if env.noise_std else agent.get_dropout_p())
        running_scores[agent_number] = 0.05 * score + (1 - 0.05) * running_scores[agent_number]

        print(f"Agent {agent_number} finished. Score {score}. Running score {running_scores[agent_number]}")

    def train_agent_steps(
        self, agent: BaseAgent, env: gym.Env, agent_results: AgentResults, iteration: int, agent_number: int,
        running_scores, num_steps):

        state = env.reset()
        self.init_states.append(state)

        scores = []
        score = 0
        for step in range(self.num_steps_per_episode):

            # env.render()

            action = agent.act(np.array(state))
            state_next, reward, done, info = env.step(action)

            score += 1

            if done:
                state_next = None

            agent.remember(state, action, reward, done, state_next)
            if not done and step == self.num_steps_per_episode - 1: #fix for A2C cause it won't reflect if not done
                num_steps[agent_number] += score
                loss, dist = agent.reflect(True, num_steps[agent_number])
            else:
                loss, dist = agent.reflect(done, num_steps[agent_number])
            agent_results.add_loss(loss, iteration, env.noise_std_ if env.noise_std else agent.get_dropout_p())
            agent_results.add_dist(dist, iteration, env.noise_std if env.noise_std else agent.get_dropout_p())

            if done:
                num_steps[agent_number] += score
                scores.append(score)
                score = 0

                state = env.reset()
                # self.init_states.append(state)

                continue

            state = state_next

        agent_results.add_score(np.mean(scores), iteration,
                                env.noise_std_dev if env.noise_std_dev else agent.get_dropout_p())
        running_scores[agent_number] = 0.05 * np.mean(scores) + (1 - 0.05) * running_scores[agent_number]

        print(f"Agent {agent_number} finished. Score {np.mean(scores)}. Running score {running_scores[agent_number]}")


    def __select_device(self, agent_number, use_cuda):
        cuda_available = torch.cuda.is_available()
        device = None
        if use_cuda and cuda_available:
            cuda_count = torch.cuda.device_count()
            device = torch.device(f"cuda:{agent_number % cuda_count}")
        else:
            device = torch.device("cpu")
        return device

