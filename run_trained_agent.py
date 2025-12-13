import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from roomba_env import RoombaEnv



def make_env():
    return RoombaEnv(world_type="carousel", gui=True)



def load_agent(
    model_path="trained_agents/dynamic_ppo_roomba_final",
    norm_path="trained_agents/dynamic_vec_normalize.pkl"
):
    env = DummyVecEnv([make_env])

    env = VecNormalize.load(norm_path, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)

    print("Loaded trained model")
    return model, env


def run(model, env, steps=100_000, sleep = .001):
    obs = env.reset()

    for step in range(steps):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, infos = env.step(action)

        #time.sleep(sleep)

        if done:
            print(f"Episode ended at step {step}, resetting...")
            obs = env.reset()


if __name__ == "__main__":
    model, env = load_agent()
    run(model, env)