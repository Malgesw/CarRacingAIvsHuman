import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

env = gym.make("CarRacing-v3", render_mode="human")
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=4)
env = VecMonitor(env, log_dir)

model = PPO.load("./checkpoints/rl_model_3500000_steps.zip", env)
evaluate_policy(model, env, n_eval_episodes=10, render=True)
