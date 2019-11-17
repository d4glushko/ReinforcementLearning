import random


class Agent:
    def __init__(self, observation_space: int, action_space: int):
        self.observation_space: int = observation_space
        self.action_space: int = action_space

    def act(self, state):
        # TODO: implement AC approach instead of random
        return random.randrange(self.action_space)

    def remember(self, state, action, reward, state_next, terminal):
        pass

    def experience_replay(self):
        pass
