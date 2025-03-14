import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

from wrappers import RewardClipper, AccBrakeMerger, GrayCropWrapper


def set_seed(seed=0):
    if seed > 0:
        np.random.seed(seed)


def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, steps, title="Learning Curve"):
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x[x <= steps], y[x <= steps])
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed (PPO)")
    plt.savefig("./images/trainingResultsPPO.png")
    plt.show()


log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
training_steps = 10000000
stackFrames = 4
clippedReward = False
# initial_lr = 2.5e-4
initial_lr = 3e-4
mergedAccBrake = False

set_seed(0)
env = gym.make("CarRacing-v3", render_mode="rgb_array")
if clippedReward:
    env = RewardClipper(env, top_clip=1.0)
if mergedAccBrake:
    env = AccBrakeMerger(env)
# env = GrayCropWrapper(env)
env = DummyVecEnv([lambda: env])
if stackFrames > 0:
    env = VecFrameStack(env, n_stack=stackFrames)
env = VecMonitor(env, log_dir)

callback = CheckpointCallback(
    save_freq=1000000,
    save_path="./checkpoints",
    name_prefix="PPOClipped{}StackedFrames{}mergedAccBrake{}".format(
        clippedReward, stackFrames, mergedAccBrake),
)

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=lambda progress_remaining: initial_lr
    * progress_remaining,  # annealing schedule for the lr
    clip_range=0.1,
    ent_coef=0.01,
    seed=0,
)

model.learn(total_timesteps=training_steps, callback=callback)
plot_results(log_dir, training_steps)
model.save(
    "./models/CarRacingAIppo{}StackedFrames{}Clipped{}".format(
        training_steps, stackFrames, clippedReward
    )
)
print("Model saved")

env.close()
