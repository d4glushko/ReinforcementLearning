import typing
import numpy as np


class Metric:
    def __init__(self, value: float, iteration: int, noise: float):
        self.value: float = value
        self.iteration: int = iteration
        self.noise: float = noise

    def to_dict(self) -> dict:
        return vars(self)

    @staticmethod
    def from_dict(metric: dict) -> 'Metric':
        return Metric(
            metric.get("value"), metric.get("iteration"), metric.get("noise")
        )


class MetricsManager:
    def __init__(self, number_of_elements: int, number_of_iterations: int):
        self.number_of_iterations: int = number_of_iterations
        self.number_of_elements: int = number_of_elements
        self.scores: typing.List[Metric] = []
        self.losses: typing.List[Metric] = []

    def add_score(self, score: float, iteration: int, noise: float):
        self.scores.append(Metric(score, iteration, noise))
        
    def add_loss(self, loss: typing.Optional[float], iteration: int, noise: float):
        if loss != None:
            self.losses.append(Metric(loss, iteration, noise))

    def get_mov_avg_scores(self) -> typing.List[Metric]:
        return self.__get_mov_avgs(self.reduce_metric(self.scores))

    def get_mov_avg_losses(self) -> typing.List[Metric]:
        return self.__get_mov_avgs(self.reduce_metric(self.losses))

    def reduce_metric(self, metrics: typing.List[Metric]) -> typing.List[Metric]:
        iterations = set([(metric.iteration, metric.noise) for metric in metrics])
        reduced_metric = [
            Metric(
                np.array([
                    value.value
                    for value in metrics
                    if value.iteration == iteration
                ]).mean(), iteration, noise
            )
            for iteration, noise in iterations
        ]
        return reduced_metric

    def __get_mov_avgs(self, metrics: typing.List[Metric]) -> typing.List[Metric]:
        avgs: typing.List[Metric] = [
            Metric(
                np.array([
                    value.value
                    for value in metrics
                    if value.iteration > metric.iteration - self.number_of_elements and 
                        value.iteration <= metric.iteration
                ]).mean(), metric.iteration, metric.noise
            )
            for metric in metrics 
            if metric.iteration % self.number_of_iterations == 0 and 
                metric.iteration >= self.number_of_elements
        ]
        return avgs
