import numpy as np

class AbstractBandit:
    def sample(self, *args):
        pass
    
    def update(self, x):
        pass


class Bandit(AbstractBandit):
    def __init__(self, mean: float = 0):
        self.mean = mean
        self.n = 0

    def sample(self, *args):
        return self.mean

    def update(self, x):
        self.n += 1
        self.mean = (1 - 1.0 / self.n) * self.mean + (1.0 / self.n) * x


class Ucb1Bandit(AbstractBandit):
    def __init__(self, mean: float = 0):
        self.mean = mean
        self.n = 0

    def sample(self, iteration):
        return self.__ucb(iteration)

    def update(self, x):
        self.n += 1
        self.mean = (1 - 1.0 / self.n) * self.mean + (1.0 / self.n) * x

    def __ucb(self, iteration):
        if self.n == 0:
            return float('inf')
        return self.mean + np.sqrt(2 * np.log(iteration) / self.n)


class BayesianBandit(AbstractBandit):
    def __init__(self):
        # parameters for mu - prior is N(0,1)
        self.m0 = 0
        self.lambda0 = 1
        self.sum_x = 0
        self.tau = 1

    def sample(self, *args):
        return np.random.randn() / np.sqrt(self.lambda0) + self.m0

    def update(self, x):
        # assume tau is 1
        self.lambda0 += 1
        self.sum_x += x
        self.m0 = self.tau * self.sum_x / self.lambda0

    @property
    def mean(self):
        return self.m0
