import typing
import numpy as np


class MetricsManager:
    def __init__(self, number_of_elements: int, number_of_iterations: int):
        self.number_of_iterations: int = number_of_iterations
        self.number_of_elements: int = number_of_elements
        self.iteration: int = 0
        self.scores: typing.List[float] = []
        self.avgs: typing.List[float] = [0]
        self.iterations: typing.List[int] = [0]

    def add_score(self, score: float):
        self.scores.append(score)
        self.iteration = self.iteration + 1
        if self.iteration % self.number_of_iterations == 0 and self.iteration >= self.number_of_elements:
            self.__cut_scores()
            avg = np.array(self.scores).mean()
            self.avgs.append(avg)
            self.iterations.append(self.iteration)

    def __cut_scores(self):
        self.scores = self.scores[-self.number_of_elements:]
