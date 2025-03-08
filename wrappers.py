import gymnasium as gym


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
