import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", render_mode="human")

state, _ = env.reset()
done = False

while not done:
    action = np.argmax(Q[state])
    state, reward, done, truncated, _ = env.step(action)

env.close()
