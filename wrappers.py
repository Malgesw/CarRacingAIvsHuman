import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
import cv2
import matplotlib.pyplot as plt


class RewardClipper(gym.RewardWrapper):
    def __init__(self, env, top_clip=None, bottom_clip=None):
        super().__init__(env)
        self.top_clip = top_clip
        self.bottom_clip = bottom_clip

    def reward(self, reward):
        if self.bottom_clip is not None:
            reward = max(reward, self.bottom_clip)
        if self.top_clip is not None:
            reward = min(reward, self.top_clip)
        return reward


class AccBrakeMerger(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = Box(low=np.array(
            [-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

    def action(self, action):
        steering = action[0]
        accBrake = action[1]

        if accBrake >= 0:
            # accelerator was pressed
            brake = 0.0
            accelerator = accBrake
        else:
            # brake was pressed
            brake = -accBrake
            accelerator = 0.0

        return np.array([steering, accelerator, brake], dtype=np.float32)


class GrayCropWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.shape

        # Update the observation space for grayscale and cropping
        cropped_height = obs_shape[0] - 12
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(cropped_height, obs_shape[1], 1), dtype=np.uint8
        )

    def observation(self, obs):
        # Convert to grayscale
        gray_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Crop bottom 12 pixels and add channel dimension
        cropped_frame = gray_frame[:-12, :][..., np.newaxis]
        return cropped_frame
