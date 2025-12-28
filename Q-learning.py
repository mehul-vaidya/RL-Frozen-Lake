import gymnasium as gym
import numpy as np

# Create Taxi environment
env = gym.make("Taxi-v3")

# State and action space
state_size = env.observation_space.n
action_size = env.action_space.n

# Q-table (agent's memory)
Q = np.zeros((state_size, action_size))

# Hyperparameters
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
episodes = 5000

# Training loop
for episode in range(episodes):
    state, _ = env.reset()
    done = False

    while not done:

        # Choose action (explore or exploit)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        # Take action
        next_state, reward, done, truncated, _ = env.step(action)

        # Q-learning update
        Q[state, action] = Q[state, action] + learning_rate * (
                reward + discount_factor * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state

    # Decay exploration rate
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

print("Training finished!")

env = gym.make("Taxi-v3", render_mode="human")

state, _ = env.reset()
done = False

while not done:
    action = np.argmax(Q[state])
    state, reward, done, truncated, _ = env.step(action)

env.close()
