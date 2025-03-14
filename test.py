import os

import gymnasium as gym

# from wrappers import RewardClipper, AccBrakeMerger, GrayCropWrapper
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

env = gym.make(
    "CarRacing-v3",
    render_mode="human",
    max_episode_steps=3000,
    lap_complete_percent=0.95,
)
# env = RewardClipper(env, top_clip=1.0)
# env = AccBrakeMerger(env)
# env = GrayCropWrapper(env)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=4)
env = VecMonitor(env, log_dir)

print(env.action_space)
print(env.observation_space)
np.random.seed(0)
# lr model 2500000 best model so far
model = PPO.load(
    "./checkpoints/PPOClippedFalseStackedFrames4DiscreteFalseAccBrakeFalse_1000000_steps.zip",
    env,
)
avg_reward, avg_std = evaluate_policy(
    model, env, n_eval_episodes=2, render=True, deterministic=False
)
print("Average reward: {} +- {}".format(avg_reward, avg_std))
