import typing
import numpy as np
import random
import os
import matplotlib
import torch
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from enum import Enum

from .utils import NoiseLearningAgents, ExchangeTypes, choose_agent
from .metrics_manager import AgentMetrics, Metrics
from .results_manager import ResultsManager, Settings, AgentResults
    

class Visualizer:
    def __init__(
        self, exchange_type: ExchangeTypes, exchange_delta: float, exchange_items_reward_count: int, agents_number: int, 
        env_name: str, noise_learning_agent: NoiseLearningAgents, metrics_number_of_elements: int, metrics_number_of_iterations: int, 
        detailed_agents_plots: bool, noise_env_step: float, noise_dropout_step: float, early_stopping: bool,
        num_steps_per_episode: int, executions_count: int, executions_from: int, execution_date: str
    ):
        self.agents_number: int = agents_number
        self.noise_learning_agent: NoiseLearningAgents = noise_learning_agent
        self.noise_env_step: float = noise_env_step
        self.noise_dropout_step = noise_dropout_step
        self.early_stopping = early_stopping
        self.env_name: str = env_name
        self.exchange_type: ExchangeTypes = exchange_type
        self.exchange_delta: float = exchange_delta
        self.exchange_items_reward_count: int = exchange_items_reward_count
        self.num_steps_per_episode = num_steps_per_episode

        self.detailed_agents_plots: bool = detailed_agents_plots
        self.metrics_number_of_elements: int = metrics_number_of_elements
        self.metrics_number_of_iterations: int = metrics_number_of_iterations

        self.executions_count: int = executions_count
        self.executions_from: int = executions_from
        self.execution_date: int = execution_date

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
                self.agents_number, self.env_name, self.noise_learning_agent.name, self.noise_env_step, self.noise_dropout_step,
                self.early_stopping, self.exchange_type.name,
                self.exchange_delta, self.exchange_items_reward_count, self.num_steps_per_episode, agent_hyper_params
            )
        )
        self.agent_metrics: typing.List[AgentMetrics] = [
            AgentMetrics() for i in range(self.agents_number)
        ]

        self.agent_play_metrics: typing.List[AgentMetrics] = [
            AgentMetrics() for i in range(self.agents_number)
        ]

    def set_train_metrics(self):
        agent_results = self.results_manager.get_train_results(self.execution_date, self.executions_count, self.executions_from)
        self.__set_metrics(agent_results, self.agent_metrics)

        self.results_number = len(agent_results)

    def set_play_metrics(self):
        agent_results = self.results_manager.get_play_results(self.execution_date, self.executions_count, self.executions_from)
        self.__set_metrics(agent_results, self.agent_play_metrics)

    def __set_metrics(self, agent_results: typing.List[typing.List[AgentResults]], agents_metrics: typing.List[AgentMetrics]):
        total = len(agent_results) * self.agents_number
        for i in range(len(agent_results)):
            for j in range(self.agents_number):
                agent_metrics = agents_metrics[j]
                agent_result = agent_results[i][j]

                agent_metrics.losses.extend(agent_result.losses)
                agent_metrics.scores.extend(agent_result.scores)
                agent_metrics.distances.extend(agent_result.distances)
                agent_metrics.exchange_attempts = agent_metrics.exchange_attempts + agent_result.exchange_attempts
                agent_metrics.exchanges = agent_metrics.exchanges + agent_result.exchanges
                print(f"Metrics set for result {i}, agent {j}. {(i * self.agents_number + j) + 1}/{total}")
                
    def show_train_metrics(self):
        self.__plot_by_noise()

        if self.detailed_agents_plots:
            self.__plot_by_agent()

        self.__plot_agent_by_noise()

        # if self.exchange_type == ExchangeTypes.RANDOM or self.exchange_type == ExchangeTypes.SMART:
        #     self.__plot_exchanges()
            
        plt.show(block=False)
        self.__save_plots()


    def show_play_metrics(self):
        self.__plot_play_agents()
        plt.show(block=False)
        self.__save_plots()


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

    def get_metrin_name_y_label(self, metric_name: str) -> str:
        mapping = {
            "scores": "Score",
            "losses": "Loss",
            "distances": "Distance"
        }
        return mapping[metric_name]

    def __plot_metrics_by_noise(self, metric_name: str):
        fig = plt.figure(figsize=(5,3.7))
        legend = []

        all_metrics = self.__get_all_metrics(metric_name)

        for noise in all_metrics.get_unique_sorted_noises():
            metrics = all_metrics.get_by_noise(noise)
            avgs = metrics.get_mov_avgs(self.metrics_number_of_elements, self.metrics_number_of_iterations)
            color = self.__get_color_by_noise(noise)
            # print(avgs.get_metric_property("iteration"))
            plt.plot(avgs.get_metric_property("iteration"), avgs.get_metric_property("value"), color=color, alpha=0.7)
            legend.append(f"Noise = {noise:.2f}")
            
        fig.suptitle(
            f"Averaged {metric_name} for {self.results_number} run(s) per Noise. "
            f"Moving average over the last {self.metrics_number_of_elements} elements every {self.metrics_number_of_iterations} iterations"
        )
        plt.ylabel(f"{self.get_metrin_name_y_label(metric_name)}")
        plt.xlabel(f"Iteration")
        plt.legend(legend, loc='best')

        print(f"Plot Metric by Noise for metric {metric_name} is ready")

    def __plot_agent_metric(self, agent_number: int, metric_name: str):
        fig = plt.figure(figsize=(5,3.7))
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
            
        fig.suptitle(
            f"Averaged {metric_name} for {self.results_number} run(s) for Agent {agent_number}. "
            f"Moving average over the last {self.metrics_number_of_elements} elements every {self.metrics_number_of_iterations} iterations"
        )
        plt.ylabel(f"{self.get_metrin_name_y_label(metric_name)}")
        plt.xlabel(f"Iteration")
        plt.legend(legend, loc='best')

        print(f"Plot Agent Metric for agent {agent_number}, metric {metric_name} is ready. Total agents: {self.agents_number}")

    def __plot_agent_by_noise(self):
        fig = plt.figure(figsize=(5,3.7))
        legend = []
        metric_name = "scores"  # can be used any metric, because we don't need values here

        for i in range(self.agents_number):
            metrics: Metrics = getattr(self.agent_metrics[i], metric_name)
            metrics = metrics.get_reduced_metrics()
            color = self.colors[i]
            plt.plot(metrics.get_metric_property("iteration"), metrics.get_metric_property("noise"), color=color, alpha=0.7)
            legend.append(f"Agent {i}")
            
        fig.suptitle(f"Agents paths for {self.results_number} run(s)")
        plt.ylabel(f"Noise")
        plt.xlabel(f"Iteration")
        plt.legend(legend, loc='best')

        print(f"Plot Agent by Noise is ready")

    def __plot_exchanges(self):
        fig = plt.figure(figsize=(5,3.7))
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

        print(f"Plot Exchanges is ready")

    def __plot_play_agents(self):
        fig = plt.figure(figsize=(5,3.7))
        legend = []
        f_path = os.path.join('diploma', 'results', self.execution_date)
        if not os.path.isdir(f_path):
            os.makedirs(f_path)
        play_scores_f = open(f_path + '/play_avg_scores.csv', 'w')
        play_scores_f.write('Agent, Avg over runs and play eps, Std,\n')

        for idx, play_metrics in enumerate(self.agent_play_metrics):
            scores_metrics = play_metrics.scores.get_reduced_metrics()
            color = self.colors[idx]
            plt.plot(scores_metrics.get_metric_property("iteration"), scores_metrics.get_metric_property("value"), color=color, alpha=0.7)
            legend.append(f"Agent {idx}")
            play_scores_f.write(','.join([str(idx), str(np.mean(scores_metrics.get_metric_property("value"))),
                                          str(np.std(scores_metrics.get_metric_property("value")))]))
            play_scores_f.write('\n')


        play_scores_f.close()
        fig.suptitle(f"Play Results. Averaged scores for {self.results_number} run(s) per Agent")
        plt.ylabel(f"Score")
        plt.xlabel(f"Iteration")
        plt.legend(legend, loc='best')

        print(f"Plot Play Agents is ready")

    def __save_plots(self):
        path_to_save = os.path.join('diploma', 'results', self.execution_date)
        if not os.path.isdir(path_to_save):
            os.makedirs(path_to_save)
        for i in plt.get_fignums():
            plt.figure(i)
            plt.savefig(os.path.join(path_to_save, f'figure_{i}.png'))

