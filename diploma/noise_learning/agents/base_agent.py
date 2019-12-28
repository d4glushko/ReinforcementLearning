import typing

from ..common.serializable import DictSerializable

class AgentHyperParams(DictSerializable):
    pass

class BaseAgent():
    agent_hyper_params: AgentHyperParams = None

    def __init__(self, observation_space: int, action_space: int, device, debug: bool):
        self.observation_space: int = observation_space
        self.action_space: int = action_space
        self.device = device
        self._debug: bool = debug

        params = {
            'observation_space': self.observation_space,
            'action_space': self.action_space,
            'device': device,
            'debug': debug
        }
        self._info_log(f"Agent Initializaed. Params: {params}")

    def act(self, state):
        if self._debug:
            self._debug_log(f"Act Started. State: {state}")

    def remember(self, state, action, reward, done, next_state):
        if self._debug:
            to_remember = {
                'state': state,
                'next_state': next_state,
                'action': action,
                'reward': reward,
                'done': done
            }
            self._debug_log(f"Remember Started. To remember: {to_remember}")
    
    def reflect(self, done) -> typing.Optional[float]:
        if self._debug:
            self._debug_log(f"Reflect Started.")
        return None

    def _debug_log(self, message):
        print(f"DEBUG. {self.__class__.__name__}. {message}")

    def _info_log(self, message):
        print(f"INFO. {self.__class__.__name__}. {message}")
