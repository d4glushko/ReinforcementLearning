import abc

class BaseAgent(abc.ABC):
    @abc.abstractmethod
    def act(self, state):
        pass

    @abc.abstractmethod
    def remember(self, state, action, reward, done, next_state):
        pass
