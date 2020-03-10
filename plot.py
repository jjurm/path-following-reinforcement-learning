import matplotlib.pylab as plt
import gym
import numpy as np

ENV_NAME = 'robot_env:robot-env-v0'

env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.shape[0]

MIN_X, MAX_X = (-5, 5)
MIN_Y, MAX_Y = (-5, 5)

def get_output(observation):
	u = 1
	w = 1
	action = [observation[0],observation[1]]
	'''
	feed observation into the agent to obtain the output
	'''
	return action

ax = plt.axes()
ax.set(xlim=(MIN_X, MAX_X), ylim=(MIN_Y, MAX_Y))

split = 30
X_grid, Y_grid = np.meshgrid(np.linspace(MIN_X, MAX_X, split),
                     np.linspace(MIN_Y, MAX_Y, split))
U = np.zeros_like(X_grid)
V = np.zeros_like(U)

for X in range(len(X_grid)):
	for Y in range (len(X_grid[0])):
		cur_X = X * (MAX_X - MIN_X) / (split-1) + MIN_X
		cur_Y = Y * (MAX_Y - MIN_Y) / (split-1) + MIN_Y
		action = get_output([cur_X, cur_Y])
		U[X, Y] = action[0]
		V[X, Y] = action[1]
		#ax.arrow(cur_X, cur_Y, cur_X, cur_Y, head_width=0.05, head_length=0.1)
ax.quiver(X_grid, Y_grid, U, V, units='width')
plt.savefig('potential_template.png')
plt.show()
		