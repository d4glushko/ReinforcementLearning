import numpy as np

class SlotMachine:
    def __init__(self, mean: float = 0):
        self.mean = mean

    def act(self):
        return np.random.randn() + self.mean