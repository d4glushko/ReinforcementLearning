import numpy as np
import matplotlib.pyplot as plt

from bandit import AbstractBandit, Bandit, BayesianBandit, Ucb1Bandit
from slot_machine import SlotMachine

class AbstractAlgorithm:
    bandit_type: type = None

    def _init_machines(self, means):
        machines = []
        for mean in means:
            machines.append(SlotMachine(mean))
        return machines

    def _init_bandits(self, n):
        bandits = []
        for i in range(n):
            bandits.append(self.bandit_type())
        return bandits

    def _is_explore(self, iteration):
        pass

    def run_experiment(self, means, n):
        machines = self._init_machines(means)
        bandits = self._init_bandits(len(means))

        results = np.empty(n)
        for i in range(n):
            iteration = i + 1
            if self._is_explore(iteration):
                machine_index = np.random.choice(len(machines))
            else:
                machine_index = np.argmax([bandit.sample(iteration) for bandit in bandits])
            res = machines[machine_index].act()
            bandits[machine_index].update(res)

            results[i] = res

        cumulative_average = np.cumsum(results) / (np.arange(n) + 1)
        plt.plot(cumulative_average)
        for mean in means:
            plt.plot(np.ones(n) * mean)
        plt.xscale('log')
        plt.show()
        
        return cumulative_average, [bandit.mean for bandit in bandits]



class EpsilonGreedy(AbstractAlgorithm):
    bandit_type = Bandit

    def __init__(self, eps):
        self.epsilon = eps

    def _is_explore(self, *args):
        p = np.random.random()
        return p < self.epsilon


class EpsilonGreedyDecay(AbstractAlgorithm):
    bandit_type = Bandit

    def _is_explore(self, iteration):
        p = np.random.random()
        return p < 1.0 / iteration


class OptimisticValue(AbstractAlgorithm):
    bandit_type = Bandit

    def __init__(self, init_mean):
        self.init_mean = init_mean

    def _init_bandits(self, n):
        bandits = []
        for i in range(n):
            bandits.append(self.bandit_type(self.init_mean))
        return bandits

    def _is_explore(self, *args):
        return False


class Ucb1(OptimisticValue):
    bandit_type = Ucb1Bandit


class ThompsonSampling(AbstractAlgorithm):
    bandit_type = BayesianBandit

    def _is_explore(self, *args):
        return False