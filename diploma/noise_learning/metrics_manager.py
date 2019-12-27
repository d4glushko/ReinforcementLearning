import typing
import numpy as np
from collections import Counter 

from .utils.serializable import DictSerializable


class Metric(DictSerializable):
    def __init__(self, value: float, iteration: int, noise: float):
        self.value: float = value
        self.iteration: int = iteration
        self.noise: float = noise


class Metrics(DictSerializable):
    def __init__(self, metrics: typing.List[Metric] = None):
        if not metrics:
            metrics = []
        self.metrics: typing.List[Metric] = metrics

    def append(self, metric: Metric):
        self.metrics.append(metric)

    def extend(self, metrics: 'Metrics'):
        self.metrics.extend(metrics.metrics)

    def get_unique_sorted_noises(self) -> typing.List[float]:
        return sorted(list(set([(metric.noise) for metric in self.metrics])))

    def get_by_noise(self, noise: float) -> 'Metrics':
        return Metrics([metric for metric in self.metrics if metric.noise == noise])

    def get_reduced_metrics(self) -> 'Metrics':
        iterations = self.get_sorted_unique_iterations()
        reduced_metrics: typing.List[Metric] = []
        for iteration in iterations:
            values = []
            noises = []
            for local_metric in self.metrics:
                if local_metric.iteration != iteration:
                    continue
                values.append(local_metric.value)
                noises.append(local_metric.noise)

            value = np.array(values).mean()
            noise = Counter(noises).most_common(1)[0][0]
            reduced_metrics.append(Metric(value, iteration, noise))
        return Metrics(reduced_metrics)

    def get_sorted_unique_iterations(self) -> typing.List[int]:
        return sorted(list(set([metric.iteration for metric in self.metrics])))

    def get_mov_avgs(self, number_of_elements: int, number_of_iterations: int) -> 'Metrics':
        metrics = self.get_reduced_metrics().metrics
        avgs: typing.List[Metric] = []
        for metric in metrics:
            if not (metric.iteration % number_of_iterations == 0 and metric.iteration >= number_of_elements):
                continue
            values = []
            noises = []
            for local_metric in metrics:
                if not (local_metric.iteration > metric.iteration - number_of_elements and local_metric.iteration <= metric.iteration):
                    continue
                values.append(local_metric.value)
                noises.append(local_metric.noise)

            value = np.array(values).mean()
            noise = Counter(noises).most_common(1)[0][0]
            avgs.append(Metric(value, metric.iteration, noise))
        return Metrics(avgs)

    def fill_noise_duplicates(self) -> 'Metrics':
        result: typing.List[Metric] = []
        metrics_len = len(self.metrics)
        for idx, metric in enumerate(self.metrics):
            result.append(metric)
            if idx + 1 < metrics_len and metric.noise != self.metrics[idx + 1].noise:
                next_metric = self.metrics[idx + 1]
                result.append(Metric(metric.value, metric.iteration, next_metric.noise))
        return Metrics(result)

    def get_metric_property(self, property_name: str) -> typing.List[typing.Any]:
        return [getattr(metric, property_name) for metric in self.metrics]

    def to_dict(self) -> dict:
        res = super().to_dict()
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
