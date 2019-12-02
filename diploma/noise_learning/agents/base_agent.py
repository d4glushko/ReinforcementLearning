class BaseAgent():
    def __init__(self, observation_space: int, action_space: int, debug: bool = False):
        self.observation_space: int = observation_space
        self.action_space: int = action_space
        self._debug: bool = debug

        if self._debug:
            params = {
                'observation_space': self.observation_space,
                'action_space': self.action_space
            }
            self._debug_log(f"Agent Initializaed. Params: {params}")

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
    
    def reflect(self):
        if self._debug:
            self._debug_log(f"Reflect Started.")

    def _debug_log(self, message):
        print(f"DEBUG. {self}. {message}")
