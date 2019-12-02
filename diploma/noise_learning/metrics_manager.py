import typing
import numpy as np


class MetricsManager:
    def __init__(self, number_of_elements: int, number_of_iterations: int):
        self.number_of_iterations: int = number_of_iterations
        self.number_of_elements: int = number_of_elements
        self.scores: typing.List[float] = []
        self.avgs: typing.List[float] = []

    def add_score(self, score: float):
        self.scores.append(score)
        if len(self.scores) % self.number_of_iterations == 0:
            avg = np.array(self.scores[-self.number_of_elements:]).mean()
            self.avgs.append(avg)
