import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from callback import RoombaEvalCallback
import numpy as np
import os

from roomba_env import RoombaEnv

def make_env():
    return RoombaEnv(world_type="carousel", gui=True)

env = DummyVecEnv([make_env])

env = VecNormalize(env, norm_obs=True, norm_reward=True)


model = PPO(
    policy="MlpPolicy",
    env=env,
    device='cuda',
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=.005,
    verbose=1
)


'''checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints/",
    name_prefix="ppo_roomba"
)'''

callback = RoombaEvalCallback(verbose=1)


model.learn(
    total_timesteps=1000000,
    callback=callback
)

model.save("ppo_roomba_final")
env.save("vec_normalize.pkl")

print("Training finished!")

np.savez(
    "roomba_training_data.npz",
    rewards=np.array(callback.episode_rewards),
    lengths=np.array(callback.episode_lengths),
    cells=np.array(callback.episode_cells),
    collisions=np.array(callback.episode_collisions),
    coverage=np.array(callback.episode_coverage)
)

