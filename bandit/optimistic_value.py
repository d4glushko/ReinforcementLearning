import numpy as np
import matplotlib.pyplot as plt

from bandit import Bandit
from slot_machine import SlotMachine

class OptimisticValue:
    @staticmethod
    def run_experiment(means, init_mean, N):
        machines = []
        bandits = []
        for mean in means:
            machines.append(SlotMachine(mean))
            bandits.append(Bandit(init_mean))
            
        results = np.empty(N)
        for i in range(N):
            machine_index = np.argmax([bandit.mean for bandit in bandits])
            res = machines[machine_index].act()
            bandits[machine_index].update(res)
            
            results[i] = res
        
        cumulative_average = np.cumsum(results) / (np.arange(N) + 1)
        plt.plot(cumulative_average)
        for mean in means:
            plt.plot(np.ones(N) * mean)
        plt.xscale('log')
        plt.show()
        
        return cumulative_average, [bandit.mean for bandit in bandits]
