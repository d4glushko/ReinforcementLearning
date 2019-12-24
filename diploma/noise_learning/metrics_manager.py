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


class Metrics:
    def __init__(self, metrics: typing.List[Metric] = None):
        if not metrics:
            metrics = []
        self.metrics: typing.List[Metric] = metrics

    def append(self, metric: Metric):
        self.metrics.append(metric)

    def get_reduced_metrics(self) -> 'Metrics':
        iterations = self.get_sorted_unique_iterations_noises()
        reduced_metrics = [
            Metric(
                np.array([
                    value.value
                    for value in self.metrics
                    if value.iteration == iteration
                ]).mean(), iteration, noise
            )
            for iteration, noise in iterations
        ]
        return Metrics(reduced_metrics)

    def get_sorted_unique_iterations_noises(self) -> typing.List[tuple]:
        return sorted(list(set([(metric.iteration, metric.noise) for metric in self.metrics])), key=lambda m: m[0])

    def get_mov_avgs(self, number_of_elements: int, number_of_iterations: int) -> 'Metrics':
        metrics = self.get_reduced_metrics()
        avgs: typing.List[Metric] = [
            Metric(
                np.array([
                    value.value
                    for value in self.metrics
                    if value.iteration > metric.iteration - number_of_elements and 
                        value.iteration <= metric.iteration
                ]).mean(), metric.iteration, metric.noise
            )
            for metric in self.metrics 
            if metric.iteration % number_of_iterations == 0 and 
                metric.iteration >= number_of_elements
        ]
        return Metrics(avgs)

    def to_dict(self) -> dict:
        res = vars(self)
        res["metrics"] = [metric.to_dict() for metric in res.get("metrics")]
        return res

    @staticmethod
    def from_dict(metrics: dict) -> 'Metrics':
        return Metrics(
            [Metric.from_dict(metric) for metric in metrics.get("metrics")]
        )


class AgentMetrics:
    def __init__(self):
        self.scores: Metrics = Metrics()
        self.losses: Metrics = Metrics()
