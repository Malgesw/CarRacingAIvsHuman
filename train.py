import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

from wrappers import RewardClipper

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
training_steps = 5000000
stackFrames = 4
clippedReward = True
initial_lr = 2.5e-4

env = gym.make("CarRacing-v3", render_mode="human")
if clippedReward:
    env = RewardClipper(env, top_clip=1.0)
env = DummyVecEnv([lambda: env])
if stackFrames > 0:
    env = VecFrameStack(env, n_stack=stackFrames)
env = VecMonitor(env, log_dir)

callback = CheckpointCallback(
    save_freq=1000000,
    save_path="./checkpoints",
    name_prefix="PPOClipped{}StackedFrames{}".format(clippedReward, stackFrames),
)

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=lambda progress_remaining: initial_lr
    * progress_remaining,  # annealing schedule for the lr
    clip_range=0.1,
    ent_coef=0.01,
)
model.learn(total_timesteps=training_steps, callback=callback)
model.save(
    "./models/CarRacingAIppo{}StackedFrames{}Clipped{}".format(
        training_steps, stackFrames, clippedReward
    )
)
print("Model saved")

env.close()
