# modified from https://github.com/alex-petrenko/curious-rl/blob/master/algorithms/env_wrappers.py
import cv2
import gym
import numpy as np

from collections import deque

from gym import spaces

def numpy_all_the_way(list_of_arrays):
    """Turn a list of numpy arrays into a 2D numpy array."""
    shape = list(list_of_arrays[0].shape)
    shape[:0] = [len(list_of_arrays)]
    arr = np.concatenate(list_of_arrays).reshape(shape)
    return arr

def unwrap_env(wrapped_env):
    return wrapped_env.unwrapped


def has_image_observations(observation_space):
    """It's a heuristic."""
    return len(observation_space.shape) >= 2


class StackFramesWrapper(gym.core.Wrapper):
    """
    Gym env wrapper to stack multiple frames.
    Useful for training feed-forward agents on dynamic games.
    """

    def __init__(self, env, stack_past_frames):
        super(StackFramesWrapper, self).__init__(env)
        if len(env.observation_space.shape) not in [1, 2]:
            raise Exception('Stack frames works with vector observations and 2D single channel images')
        self._stack_past = stack_past_frames
        self._frames = None

        self._image_obs = has_image_observations(env.observation_space)

        if self._image_obs:
            new_obs_space_shape = (stack_past_frames,) + env.observation_space.shape
        else:
            new_obs_space_shape = list(env.observation_space.shape)
            new_obs_space_shape[0] *= stack_past_frames

        self.observation_space = spaces.Box(
            0.0 if self._image_obs else env.observation_space.low[0],
            1.0 if self._image_obs else env.observation_space.high[0],
            shape=new_obs_space_shape,
            dtype=env.observation_space.dtype,
        )

    def _render_stacked_frames(self):
        if self._image_obs:
            return numpy_all_the_way(self._frames)
        else:
            return np.array(self._frames).flatten()

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self._frames = deque([observation] * (self._stack_past))
        return self._render_stacked_frames()

    def step(self, action):
        new_observation, reward, done, info = self.env.step(action)
        self._frames.popleft()
        self._frames.append(new_observation)
        return self._render_stacked_frames(), reward, done, info

    def get_obs(self):
        return self._render_stacked_frames()


class ResizeAndGrayscaleWrapper(gym.core.Wrapper):
    """Resize observation frames to specified (w,h) and convert to grayscale."""

    def __init__(self, env, w, h):
        super(ResizeAndGrayscaleWrapper, self).__init__(env)
        self.observation_space = spaces.Box(0, 255, shape=[w, h], dtype=np.uint8)
        self.w = w
        self.h = h

    def _observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.w, self.h), interpolation=cv2.INTER_AREA)
        # obs = obs.astype(np.float32) / 255.0  # comment to save memory if using replay buffer
        return obs

    def reset(self, **kwargs):
        return self._observation(self.env.reset(**kwargs))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._observation(obs), reward, done, info

    def get_obs(self):
        return self._observation(self.env.get_obs())

class ResizeWrapper(gym.core.Wrapper):
    """Resize observation frames to specified (w,h)"""

    def __init__(self, env, w, h):
        super(ResizeWrapper, self).__init__(env)
        self.observation_space = spaces.Box(0, 255, shape=[3, w, h], dtype=np.uint8)
        self.w = w
        self.h = h

    def _observation(self, obs):
        obs = cv2.resize(obs, (self.w, self.h), interpolation=cv2.INTER_AREA)
        obs = obs.transpose((2, 0, 1))
        return obs

    def reset(self, **kwargs):
        return self._observation(self.env.reset(**kwargs))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._observation(obs), reward, done, info

    def get_obs(self):
        return self._observation(self.env.get_obs())

