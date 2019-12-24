import typing
import numpy as np
import random
import matplotlib
import torch
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from enum import Enum

from .metrics_manager import MetricsManager, Metric
from .results_manager import ResultsManager, Settings


class NoiseLearningAgents(Enum):
    DQN = 1
    A2C = 2
    

class Visualizer:
    def __init__(
        self, agents_number: int, env_name: str, noise_learning_agent: NoiseLearningAgents, 
        metrics_number_of_elements: int, metrics_number_of_iterations: int, noise_env_step: float
    ):
        self.agents_number: int = agents_number
        self.noise_learning_agent: NoiseLearningAgents = noise_learning_agent
        self.noise_env_step: float = noise_env_step
        self.env_name: str = env_name
        self.metrics_number_of_elements: int = metrics_number_of_elements
        self.metrics_number_of_iterations: int = metrics_number_of_iterations
        self.noise_colors: dict = {}

        self.results_number: int = 1

        self.__setup_metrics()

    def __setup_metrics(self):
        self.metrics: typing.List[MetricsManager] = [
            MetricsManager(self.metrics_number_of_elements, self.metrics_number_of_iterations) for i in range(self.agents_number)
        ]

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
        # self.__plot_losses()
        plt.show(block=False)

    def __get_result_settings(self) -> Settings:
        return Settings(
            self.agents_number, self.env_name, self.noise_learning_agent.name, self.noise_env_step
        )

    def __plot_scores(self):
        fig = plt.figure()
        legend = []
        for i in range(self.agents_number):
            metrics = self.metrics[i]
            avgs: typing.List[Metric] = metrics.get_mov_avg_scores()
            iterations = np.array([avg.iteration for avg in avgs])
            values = np.array([avg.value for avg in avgs])
            noises = np.array([avg.noise for avg in avgs])
            noises[10:] = 1

            # noise_1_y = np.ma.masked_where(noises < 0.05, values)
            # noise_2_y = np.ma.masked_where(noises >= 0.05, values)


            name = "Set3"
            cmap = get_cmap(name)
            colors = ['red', 'blue']

            for noise in np.unique(noises):
                color = self.noise_colors.get(noise)
                if not color:
                    color = colors[len(self.noise_colors)]
                    self.noise_colors[noise] = color

                y = np.ma.masked_where(noises != noise, values)
                plt.plot(iterations, y, color=color)
                legend.append(f"Noise = {noise:.2f}")

            # plt.plot(iterations, noise_1_y, color=(1,1,0,1))
            # plt.plot(iterations, noise_2_y, color=(0,1,1,1))
            # legend.append(f"Agent {i}, Current Noise = {noise:.2f}")
            
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
            colors = []
            for avg in avgs:
                if avg.noise < 0.05:
                    colors.append((255,0,0,1))
                else:
                    colors.append((0,0,255,1))
            plt.plot([avg.iteration for avg in avgs], [avg.value for avg in avgs], color=colors)
            legend.append(f"Agent {i}, Current Noise = {noise:.2f}")
            
        fig.suptitle(f"Averaged Loss for {self.results_number} run(s)")
        plt.ylabel(f"Moving avg over the last {metrics.number_of_elements} elements every {metrics.number_of_iterations} iterations")
        plt.xlabel(f"Env Iterations")
        plt.legend(legend, loc='upper left')
