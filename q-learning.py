import numpy as np

import gym

# Loading the environment for FrozenLake
env = gym.make('FrozenLake-v0')

# Implementing Q-Table learning algorithm

# Initialize the Q-Table. In this case we have a 16x4 table. (16 possible states and 4 possible actions)
Q = np.zeros([env.observation_space.n, env.action_space.n])

# The learning rate
lr = 0.2

# The discount factor
y = 0.8

# The number of episodes
n_episodes = 1000

# List of all rewards
rewards = []

# Initialize empty list to contain total rewards
for i in range(n_episodes):
	state = env.reset()
	reward_all = 0
	done = False

	# The Q-Table learning algorithm
	for j in range(100):

		# Using greedy action choosing with some noise
		action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n)*(1/(i + 1)))

		# Get new state and reward from environment
		state_new, reward, done, _ = env.step(action)

		# Update the Q-table from learnt knowledge
		Q[state, action] = Q[state, action] + lr * (reward + y * np.max(Q[state_new, :]) - Q[state, action])
		
		# Accumulate the rewards 		
		reward_all = reward_all + reward

		state = state_new

		if done == True:
			break
		
	rewards.append(reward_all)

print("Score: %f"%(sum(rewards)/n_episodes))
print('\nFinal Q-Table:\n')
print(Q)
