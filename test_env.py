from turtlebot_env import TurtleBotEnv
import numpy as np

env = TurtleBotEnv(render_mode="human")

obs, info = env.reset()

for _ in range(300):
    action = np.array([0.2, 0.3])
    obs, r, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

env.close()
