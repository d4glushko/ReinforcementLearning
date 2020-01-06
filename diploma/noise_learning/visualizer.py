import typing
import numpy as np
import random
import matplotlib
import torch
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from enum import Enum

from .utils import NoiseLearningAgents, ExchangeTypes, choose_agent
from .metrics_manager import AgentMetrics, Metrics
from .results_manager import ResultsManager, Settings
    

class Visualizer:
    def __init__(
        self, exchange_type: ExchangeTypes, exchange_delta: float, exchange_items_reward_count: int, agents_number: int, 
        env_name: str, noise_learning_agent: NoiseLearningAgents, metrics_number_of_elements: int, metrics_number_of_iterations: int, 
        detailed_agents_plots: bool, noise_env_step: float, executions_count: int, executions_from: int, execution_date:str, execution_number:str
    ):
        self.agents_number: int = agents_number
        self.noise_learning_agent: NoiseLearningAgents = noise_learning_agent
        self.noise_env_step: float = noise_env_step
        self.env_name: str = env_name
        self.exchange_type: ExchangeTypes = exchange_type
        self.exchange_delta: float = exchange_delta
        self.exchange_items_reward_count: int = exchange_items_reward_count

        self.detailed_agents_plots: bool = detailed_agents_plots
        self.metrics_number_of_elements: int = metrics_number_of_elements
        self.metrics_number_of_iterations: int = metrics_number_of_iterations

        self.executions_count: int = executions_count
        self.executions_from: int = executions_from
        self.execution_date: str = execution_date
        self.execution_number: str = execution_number

        self.noise_colors: dict = {}
        self.results_number: int = 1

        self.__setup_metrics()
        self.__setup_colors()

    def __setup_colors(self):
        name = "Paired"
        cmap = get_cmap(name)
        self.colors = cmap.colors

    def __setup_metrics(self):
        agent_hyper_params = choose_agent(self.noise_learning_agent).agent_hyper_params.to_dict()
        self.results_manager: ResultsManager = ResultsManager(
            Settings(
                self.agents_number, self.env_name, self.noise_learning_agent.name, self.noise_env_step, self.exchange_type.name, 
                self.exchange_delta, self.exchange_items_reward_count, agent_hyper_params
            ),
        self.execution_date, self.execution_number)
        self.agent_metrics: typing.List[AgentMetrics] = [
            AgentMetrics() for i in range(self.agents_number)
        ]

    def set_metrics(self):
        agent_results = self.results_manager.get_results(self.executions_count, self.executions_from)
        for i in range(len(agent_results)):
            for j in range(self.agents_number):
                agent_metrics = self.agent_metrics[j]
                agent_result = agent_results[i][j]

                agent_metrics.losses.extend(agent_result.losses)
                agent_metrics.scores.extend(agent_result.scores)
                agent_metrics.distances.extend(agent_result.distances)
                agent_metrics.exchange_attempts = agent_metrics.exchange_attempts + agent_result.exchange_attempts
                agent_metrics.exchanges = agent_metrics.exchanges + agent_result.exchanges


        self.results_number = len(agent_results)
                
    def show_metrics(self):
        self.__plot_by_noise()

        if self.detailed_agents_plots:
            self.__plot_by_agent()

        self.__plot_agent_by_noise()

        if self.exchange_type == ExchangeTypes.RANDOM or self.exchange_type == ExchangeTypes.SMART:
            self.__plot_exchanges()

        plt.show(block=False)

    def __get_all_metrics(self, metric_name: str) -> Metrics:
        all_metrics: Metrics = Metrics()
        for i in range(self.agents_number):
            all_metrics.extend(getattr(self.agent_metrics[i], metric_name))
        return all_metrics

    def __get_color_by_noise(self, noise: float):
        color = self.noise_colors.get(noise)
        if not color:
            color = self.colors[len(self.noise_colors)]
            self.noise_colors[noise] = color
        return color

    def __plot_by_noise(self):
        self.__plot_metrics_by_noise("scores")
        self.__plot_metrics_by_noise("losses")
        self.__plot_metrics_by_noise("distances")

    def __plot_by_agent(self):
        for i in range(self.agents_number):
            self.__plot_agent_metric(i, "scores")
            self.__plot_agent_metric(i, "losses")
            self.__plot_agent_metric(i, "distances")

    def __plot_metrics_by_noise(self, metric_name: str):
        fig = plt.figure()
        legend = []

        all_metrics = self.__get_all_metrics(metric_name)

        for noise in all_metrics.get_unique_sorted_noises():
            metrics = all_metrics.get_by_noise(noise)
            avgs = metrics.get_mov_avgs(self.metrics_number_of_elements, self.metrics_number_of_iterations)
            color = self.__get_color_by_noise(noise)
            plt.plot(avgs.get_metric_property("iteration"), avgs.get_metric_property("value"), color=color)
            legend.append(f"Noise = {noise:.2f}")
            
        fig.suptitle(f"Averaged {metric_name} for {self.results_number} run(s) per Noise")
        plt.ylabel(f"Moving avg over the last {self.metrics_number_of_elements} elements every {self.metrics_number_of_iterations} iterations")
        plt.xlabel(f"Iterations")
        plt.legend(legend, loc='upper left')

    def __plot_agent_metric(self, agent_number: int, metric_name: str):
        fig = plt.figure()
        legend = []
        
        metrics: Metrics = getattr(self.agent_metrics[agent_number], metric_name)
        avgs: Metrics = metrics.get_mov_avgs(self.metrics_number_of_elements, self.metrics_number_of_iterations)

        avgs_with_noise_dups = avgs.fill_noise_duplicates()
        iterations = np.array(avgs_with_noise_dups.get_metric_property("iteration"))
        values = np.array(avgs_with_noise_dups.get_metric_property("value"))
        noises = np.array(avgs_with_noise_dups.get_metric_property("noise"))
        for noise in avgs_with_noise_dups.get_unique_sorted_noises():
            color = self.__get_color_by_noise(noise)
            y = np.ma.masked_where(noises != noise, values)
            plt.plot(iterations, y, color=color)
            legend.append(f"Noise = {noise:.2f}")
            
        fig.suptitle(f"Averaged {metric_name} for {self.results_number} run(s) for Agent {agent_number}")
        plt.ylabel(f"Moving avg over the last {self.metrics_number_of_elements} elements every {self.metrics_number_of_iterations} iterations")
        plt.xlabel(f"Iterations")
        plt.legend(legend, loc='upper left')

    def __plot_agent_by_noise(self):
        fig = plt.figure()
        legend = []
        metric_name = "scores"  # can be used any metric, because we don't need values here

        for i in range(self.agents_number):
            metrics: Metrics = getattr(self.agent_metrics[i], metric_name)
            metrics = metrics.get_reduced_metrics()
            plt.plot(metrics.get_metric_property("iteration"), metrics.get_metric_property("noise"))
            legend.append(f"Agent {i}")
            
        fig.suptitle(f"Agents pathes for {self.results_number} run(s)")
        plt.ylabel(f"Noise")
        plt.xlabel(f"Iterations")
        plt.legend(legend, loc='upper left')

    def __plot_exchanges(self):
        fig = plt.figure()
        x_labels = []
        y_pos = []
        values = []
        total_exchanges = 0
        total_exchange_attempts = 0

        for i in range(self.agents_number):
            agent_metrics = self.agent_metrics[i]
            x_labels.append(f"Agent {i}")
            y_pos.append(i)
            values.append(agent_metrics.exchanges / agent_metrics.exchange_attempts)
            total_exchanges = total_exchanges + agent_metrics.exchanges
            total_exchange_attempts = total_exchange_attempts + agent_metrics.exchange_attempts
        
        plt.bar(y_pos, values, align="center", alpha=0.5)
        plt.xticks(y_pos, x_labels)
            
        fig.suptitle(f"Agents exchange rates. Total rate: {total_exchanges / total_exchange_attempts:.2f}, total attempts: {total_exchange_attempts}")
        plt.ylabel(f"Exchanges / Attempts Rate")
        plt.xlabel(f"Agents")
