import typing
import numpy as np
import matplotlib
import torch
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from enum import Enum

from .agents.base_agent import BaseAgent
from .agents.a2c_agent import A2CAgent
from .agents.dqn_agent import DqnAgent
from .envs.env import EnvironmentWrapper
from .metrics_manager import MetricsManager, Metric
from .results_manager import ResultsManager, Settings, AgentResults


class NoiseLearningAgents(Enum):
    DQN = 1
    A2C = 2
    

class NoiseLearning:
    def __init__(
        self, training_episodes: int, agents_number: int, env_name: str, noise_learning_agent: NoiseLearningAgents, debug: bool, 
        metrics_number_of_elements: int, metrics_number_of_iterations: int, noise_env_step: float, use_cuda: bool, 
        current_execution: int = 1, total_executions: int = 1, ignore_training_setup: bool = False
    ):
        self.training_episodes: int = training_episodes
        self.agents_number: int = agents_number
        self.noise_learning_agent: NoiseLearningAgents = noise_learning_agent
        self.noise_env_step: float = noise_env_step
        self.env_name: str = env_name
        self.use_cuda: bool = use_cuda
        self.debug: bool = debug
        self.metrics_number_of_elements: int = metrics_number_of_elements
        self.metrics_number_of_iterations: int = metrics_number_of_iterations

        self.current_execution: int = current_execution
        self.total_executions: int = total_executions

        self.results_number: int = 1

        self.__setup_envs()
        if not ignore_training_setup:
            self.__setup_agents()
        self.__setup_metrics()

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

    def __setup_metrics(self):
        self.metrics: typing.List[MetricsManager] = [
            MetricsManager(self.metrics_number_of_elements, self.metrics_number_of_iterations) for i in range(self.agents_number)
        ]

    def __choose_agent(self, noise_learning_agent: NoiseLearningAgents) -> typing.Type[BaseAgent]:
        agents_mapping = {
            NoiseLearningAgents.DQN: DqnAgent,
            NoiseLearningAgents.A2C: A2CAgent
        }
        return agents_mapping[noise_learning_agent]

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
                metrics = self.metrics[j]

                self.__train_agent_episode(agent, env, metrics, i, j)

            if self.should_swap_agents():
                self.swap_agents()

    def __select_device(self, agent_number, use_cuda):
        cuda_available = torch.cuda.is_available()
        device = None
        if use_cuda and cuda_available:
            cuda_count = torch.cuda.device_count()
            device = torch.device(f"cuda:{agent_number % cuda_count}")
        else:
            device = torch.device("cpu")
        return device

    def __train_agent_episode(
        self, agent: BaseAgent, env: EnvironmentWrapper, metrics: MetricsManager, iteration: int, agent_number: int
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
            metrics.add_loss(agent.last_loss, iteration)

            if done:
                break

            state = state_next
        metrics.add_score(score, iteration)
        print(f"Agent {agent_number} finished. Score {score}")

    def set_metrics(self):
        settings = self.__get_result_settings()
        agent_results = ResultsManager().get_results(settings)
        for i in range(len(agent_results)):
            for j in range(self.agents_number):
                agent_result = agent_results[i][j]
                self.metrics[j].losses.extend(agent_result.losses)
                self.metrics[j].scores.extend(agent_result.scores)

        self.results_number = len(agent_results)
                
    def show_metrics(self):
        self.__plot_scores()
        self.__plot_losses()
        plt.show(block=False)

    def save_results(self):
        settings = self.__get_result_settings()

        agent_results = [
            AgentResults(metrics.reduce_metric(metrics.scores), metrics.reduce_metric(metrics.losses)) for metrics in self.metrics
        ]
        ResultsManager().save_results(settings, agent_results)

    def __get_result_settings(self) -> Settings:
        return Settings(
            self.agents_number, self.env_name, self.noise_learning_agent.name, self.noise_env_step
        )

    def __plot_scores(self):
        fig = plt.figure()
        legend = []
        for i in range(self.agents_number):
            metrics = self.metrics[i]
            noise = self.environments[i].noise_std_dev
            avgs: typing.List[Metric] = metrics.get_mov_avg_scores()
            plt.plot([avg.iteration for avg in avgs], [avg.value for avg in avgs])
            legend.append(f"Agent {i}, Current Noise = {noise:.2f}")
            
        fig.suptitle(f"Averaged Score for {self.results_number} run(s)")
        plt.ylabel(f"Moving avg over the last {metrics.number_of_elements} elements every {metrics.number_of_iterations} iterations")
        plt.xlabel(f"Env Iterations")
        plt.legend(legend, loc='upper left')
    
    def __plot_losses(self):
        fig = plt.figure()
        legend = []
        for i in range(self.agents_number):
            metrics = self.metrics[i]
            noise = self.environments[i].noise_std_dev
            avgs: typing.List[Metric] = metrics.get_mov_avg_losses()
            plt.plot([avg.iteration for avg in avgs], [avg.value for avg in avgs])
            legend.append(f"Agent {i}, Current Noise = {noise:.2f}")
            
        fig.suptitle(f"Averaged Loss for {self.results_number} run(s)")
        plt.ylabel(f"Moving avg over the last {metrics.number_of_elements} elements every {metrics.number_of_iterations} iterations")
        plt.xlabel(f"Env Iterations")
        plt.legend(legend, loc='upper left')

    def should_swap_agents(self):
        pass

    def swap_agents(self):
        pass
