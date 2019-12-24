import typing
import os
import time
import json

from .metrics_manager import Metric, Metrics

class Settings:
    def __init__(self, agents_number: int, env_name: str, noise_learning_agent: str, noise_env_step: float):
        self.agents_number: int = agents_number
        self.env_name: str = env_name
        self.noise_learning_agent: str = noise_learning_agent
        self.noise_env_step: float = noise_env_step

    def is_same_settings(self, settings: 'Settings'):
        return self.agents_number == settings.agents_number and \
                self.env_name == settings.env_name and \
                self.noise_learning_agent == settings.noise_learning_agent and \
                self.noise_env_step == settings.noise_env_step

    def to_dict(self) -> dict:
        return vars(self)

    @staticmethod
    def from_dict(settings: dict) -> 'Settings':
        return Settings(
            settings.get("agents_number"), settings.get("env_name"), settings.get("noise_learning_agent"), 
            settings.get("noise_env_step")
        )


class AgentResults:
    def __init__(self):
        self.scores: Metrics = Metrics()
        self.losses: Metrics = Metrics()

    def add_score(self, score: float, iteration: int, noise: float):
        self.scores.append(Metric(score, iteration, noise))
        
    def add_loss(self, loss: typing.Optional[float], iteration: int, noise: float):
        if loss != None:
            self.losses.append(Metric(loss, iteration, noise))

    def reduce_results(self):
        self.scores = self.scores.get_reduced_metrics()
        self.losses = self.losses.get_reduced_metrics()

    def to_dict(self) -> dict:
        self.reduce_results()
        res = vars(self)
        res["scores"] = res.get("scores").to_dict()
        res["losses"] = res.get("losses").to_dict()
        return res

    @staticmethod
    def from_dict(results: dict) -> 'AgentResults':
        agent_results = AgentResults()
        agent_results.scores = Metrics.from_dict(results.get('scores'))
        agent_results.losses = Metrics.from_dict(results.get('losses'))
        return agent_results



class ResultsManager:
    results_path = ["diploma", "temp_results"]
    settings_filename = "settings.txt"
    agent_filename = "agent{}.txt"

    def __init__(self, settings: Settings):
        self.settings: Settings = settings

    def save_results(self, agents_results: typing.List[AgentResults]):
        now = str(int(time.time()))
        target_path = os.path.join(*self.results_path, now)
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        settings_file_path = os.path.join(target_path, self.settings_filename)
        self.__save_json(settings_file_path, self.settings.to_dict())

        for i in range(self.settings.agents_number):
            agent_results = agents_results[i]
            agent_file_path = os.path.join(target_path, self.agent_filename.format(i))
            self.__save_json(agent_file_path, agent_results.to_dict())

    def get_results(self) -> typing.List[typing.List[AgentResults]]:
        agents_results: typing.List[typing.List[AgentResults]] = []
        source_path = os.path.join(*self.results_path)
        for f in os.scandir(source_path):
            if not f.is_dir():
                continue

            result_dir = f.path
            settings_file_path = os.path.join(result_dir, self.settings_filename)
            agent_settings = Settings.from_dict(self.__get_json(settings_file_path))
            if not self.settings.is_same_settings(agent_settings):
                continue
            
            current_agents_results: typing.List[AgentResults] = []
            for i in range(self.settings.agents_number):
                agent_file_path = os.path.join(result_dir, self.agent_filename.format(i))
                current_agents_results.append(AgentResults.from_dict(self.__get_json(agent_file_path)))
            
            agents_results.append(current_agents_results)
            
        return agents_results

    def __get_json(self, file_path: str) -> dict:
        with open(file_path) as json_file:
            data = json.load(json_file)
        return data


    def __save_json(self, file_path: str, data: dict):
        with open(file_path, 'w') as outfile:
            json.dump(data, outfile)
