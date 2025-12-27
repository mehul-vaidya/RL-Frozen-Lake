import gymnasium as gym
import numpy as np

# Create FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=True)

# Get state and action sizes
state_size = env.observation_space.n
action_size = env.action_space.n
print(state_size)
print(action_size)

# Create Q-table (dog's notebook)
Q = np.zeros((state_size, action_size))

# Hyperparameters
learning_rate = 0.1      # how fast dog learns
discount_factor = 0.99   # care about future treats
epsilon = 1.0            # exploration rate
epsilon_decay = 0.995
min_epsilon = 0.01
episodes = 2000

# Training loop
for episode in range(episodes):
    state, _ = env.reset()
    done = False

    while not done:

        # Exploration vs Exploitation
        #first it will explore more then slowly we will reduce epsilon
        #so that it will start exploiting
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # explore
            print("explore")
        else:
            action = np.argmax(Q[state])        # exploit
            print("exploit")

        # Take action
        next_state, reward, done, truncated, _ = env.step(action)

        # Q-learning update
        Q[state, action] = Q[state, action] + learning_rate * (
                reward + discount_factor * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state

    # Reduce exploration
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

print("Training finished!")

env = gym.make("FrozenLake-v1", render_mode="human")

state, _ = env.reset()
done = False

while not done:
    action = np.argmax(Q[state]) #From everything I’ve learned, what is the BEST move right now?”
    '''
    “Dog makes a move, environment reacts.”
    After action: 
    state → dog’s new position 
    reward → did dog get a treat?
    done → did training end?
    truncated → time limit reached (ignore for now)
    Example:
    Dog moves right
    Slips into hole
    Reward = 0
    done = True ❌
    OR
    Dog reaches goal
    Reward = 1
    done = True 🎉
    '''

    state, reward, done, truncated, _ = env.step(action)

env.close()

