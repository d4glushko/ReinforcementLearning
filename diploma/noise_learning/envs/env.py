import gym
import numpy as np
from enum import Enum
from collections import deque
import cv2


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        h = env.observation_space.high.max()

        self.observation_space = gym.spaces.Box(low=0, high=h, shape=(shp[:-1] + (shp[-1] * k, )), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))



class EnvironmentWrapper:
    def __init__(self, env_name: str, noise_std_dev: float = 0):
        self.env = gym.make(env_name)
        self.noise_std_dev: float = noise_std_dev
        self.is_noise: bool = not (self.noise_std_dev == 0)

    def observation_space(self):
        raise NotImplementedError()

    def action_space(self):
        return self.env.action_space.n

    def reset(self):
        raise NotImplementedError

    def render(self):
        return self.env.render()

    def step(self, action):
       raise NotImplementedError()

    def __noised_state(self, state):
        raise NotImplementedError()

    def __sample_scale_noise(self) -> float:
        raise NotImplementedError()


class AddCartPoleNoise(gym.Wrapper):
    def __init__(self, env: gym.Env, noise_std_dev: float = 0):
        gym.Wrapper.__init__(self, env)
        self.noise_std = noise_std_dev

    # def observation_space(self):
    #     return self.env.observation_space.shape[0]

    def reset(self):
        state = self.env.reset()
        state = self.__noised_state(state)
        return state

    def __noised_state(self, state):
        return state * self.__sample_scale_noise()

    def __sample_scale_noise(self) -> float:
        mean = 1
        noise = mean
        if self.is_noise:
            noise = np.random.normal(mean, self.noise_std)
        return noise

    def step(self, action):
        state_next, reward, done, info = self.env.step(action)
        if done:
            reward = -200
        state_next = self.__noised_state(state_next)
        return state_next, reward, done, info


class FrameProcess(gym.Wrapper):
    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(84, 84, 3),
                                                dtype=env.observation_space.dtype)


    # def observation_space(self):
    #     return self.env.observation_space.shape[2]

    def __process_frame(self, frame):
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        frame = np.divide(frame, 255)
        return frame

    def reset(self):
        state = self.env.reset()
        state = self.__process_frame(state)
        return state

    def step(self, action):
        state_next, reward, done, info = self.env.step(action)
        state_next = self.__process_frame(state_next)
        return state_next, reward, done, info


class AddAtariNoise(gym.Wrapper):
    def __init__(self, env: gym.Env,  noise_std_dev: float = 0):
        gym.Wrapper.__init__(self, env)
        self.noise_std = noise_std_dev

    def __noised_state(self, state):
        return state + self.__sample_scale_noise()

    def __sample_scale_noise(self) -> float:
        row, col, ch = self.env.observation_space.shape
        mean = 0
        gauss = np.random.normal(mean, self.noise_std, (row, col, ch))
        return gauss

    def reset(self):
        state = self.env.reset()
        state = self.__noised_state(state)
        return state

    def step(self, action):
        state_next, reward, done, info = self.env.step(action)
        state_next = self.__noised_state(state_next)
        return state_next, reward, done, info


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]
