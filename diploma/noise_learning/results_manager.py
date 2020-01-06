import typing
import os
import time
import datetime
import json

from .metrics_manager import Metric, Metrics
from .common.serializable import DictSerializable
from .agents.base_agent import AgentHyperParams
from .utils import ExchangeTypes


class Settings(DictSerializable):
    def __init__(
        self, agents_number: int, env_name: str, noise_learning_agent: str, noise_env_step: float, exchange_type: str, 
        exchange_delta: float, exchange_items_reward_count: int, agent_hyper_params: dict
    ):
        self.agents_number: int = agents_number
        self.env_name: str = env_name
        self.noise_learning_agent: str = noise_learning_agent
        self.noise_env_step: float = noise_env_step
        self.exchange_type: str = exchange_type
        self.exchange_delta: float = exchange_delta
        self.exchange_items_reward_count: int = exchange_items_reward_count
        self.agent_hyper_params: dict = agent_hyper_params

    def is_same_settings(self, settings: 'Settings'):
        result = self.agents_number == settings.agents_number and \
                self.env_name == settings.env_name and \
                self.noise_learning_agent == settings.noise_learning_agent and \
                self.noise_env_step == settings.noise_env_step and \
                self.exchange_type == settings.exchange_type and \
                self.agent_hyper_params == settings.agent_hyper_params

        if self.exchange_type == ExchangeTypes.SMART.name:
            result = result and \
                self.exchange_delta == settings.exchange_delta and \
                self.exchange_items_reward_count == settings.exchange_items_reward_count

        return result


class AgentResults(DictSerializable):
    def __init__(self):
        self.scores: Metrics = Metrics()
        self.losses: Metrics = Metrics()
        self.distances: Metrics = Metrics()
        self.exchange_attempts: int = 0
        self.exchanges: int = 0

    def add_score(self, score: float, iteration: int, noise: float):
        self.scores.append(Metric(score, iteration, noise))
        
    def add_loss(self, loss: typing.Optional[float], iteration: int, noise: float):
        if loss != None:
            self.losses.append(Metric(loss, iteration, noise))

    def add_dist(self, dist: typing.Optional[float], iteration: int, noise: float):
        if dist != None:
            self.distances.append(Metric(dist, iteration, noise))

    def reduce_results(self):
        self.scores = self.scores.get_reduced_metrics()
        self.losses = self.losses.get_reduced_metrics()
        self.distances = self.distances.get_reduced_metrics()

    def to_dict(self) -> dict:
        self.reduce_results()
        res = super().to_dict()
        res["scores"] = res.get("scores").to_dict()
        res["losses"] = res.get("losses").to_dict()
        res["distances"] = res.get("distances").to_dict()
        return res

    @staticmethod
    def from_dict(results: dict) -> 'AgentResults':
        agent_results = AgentResults()
        agent_results.scores = Metrics.from_dict(results.get('scores'))
        agent_results.losses = Metrics.from_dict(results.get('losses'))
        agent_results.distances = Metrics.from_dict(results.get('distances'))
        agent_results.exchange_attempts = results.get('exchange_attempts', 0)
        agent_results.exchanges = results.get('exchanges', 0)
        return agent_results



class ResultsManager:
    results_path = ["diploma", "temp_results", ""]
    settings_filename = "settings.txt"
    agent_filename = "agent{}.txt"

    def __init__(self, settings: Settings, execution_date: str, execution_number: str):
        self.settings: Settings = settings
        self.execution_date: str = execution_date
        self.execution_number: str = execution_number

    def save_results(self, agents_results: typing.List[AgentResults]):
        now = datetime.datetime.utcfromtimestamp(float(self.execution_date)).strftime('%Y-%m-%d_%H:%M:%S')
        target_dir = f"{now}_{self.settings.noise_learning_agent}_{self.settings.exchange_type}_{self.execution_number}"
        target_path = os.path.join(*self.results_path, target_dir)
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        settings_file_path = os.path.join(target_path, self.settings_filename)
        self.__save_dict(settings_file_path, self.settings.to_dict())

        for i in range(self.settings.agents_number):
            agent_results = agents_results[i]
            agent_file_path = os.path.join(target_path, self.agent_filename.format(i))
            self.__save_dict(agent_file_path, agent_results.to_dict())

    def get_results(self, executions_count: int = None, executions_from: int = None) -> typing.List[typing.List[AgentResults]]:
        if not executions_from:
            executions_from = 0

        agents_results: typing.List[typing.List[AgentResults]] = []
        source_path = os.path.join(*self.results_path)

        counter = -1
        for f in sorted(os.scandir(source_path), key=lambda x: x.path):
            if not f.is_dir():
                continue

            result_dir = f.path

            settings_file_path = os.path.join(result_dir, self.settings_filename)
            agent_settings = Settings.from_dict(self.__get_dict(settings_file_path))

            # if not self.settings.is_same_settings(agent_settings):
            #     continue

            counter = counter + 1
            if counter < executions_from:
                continue
            if executions_count and (counter >= executions_count + executions_from):
                break
            print(result_dir)
            current_agents_results: typing.List[AgentResults] = []
            for i in range(self.settings.agents_number):
                agent_file_path = os.path.join(result_dir, self.agent_filename.format(i))
                current_agents_results.append(AgentResults.from_dict(self.__get_dict(agent_file_path)))
            
            agents_results.append(current_agents_results)
            
        return agents_results

    def __get_dict(self, file_path: str) -> dict:
        with open(file_path) as json_file:
            data = json.load(json_file)
        return data


    def __save_dict(self, file_path: str, data: dict):
        with open(file_path, 'w') as outfile:
            json.dump(data, outfile)
