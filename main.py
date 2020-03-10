import warnings
warnings.filterwarnings('ignore')

import numpy as np
import gym

from keras.layers import Input
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from models import actor, critic

############
#   Setup
############
ENV_NAME = 'robot_env:robot-env-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.shape[0]

################
#   Parameters
################
# Training parameters
<<<<<<< HEAD
batch_size = 64
lr = 1e-4
max_episode_steps = 100
limit = 100000
nb_steps = 10000

=======
batch_size = 8
lr = 1e-3
max_episode_steps = 150
>>>>>>> 484bf4449eb0aada0173d026314e91633a6dcc8a
# Agent parameters
num_steps_warmup_critic = 100
num_steps_warmup_actor = 100
gamma = 0.99

###############
#    Models
###############
action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(batch_size,) + env.observation_space.shape, name='observation_input')
num_feat = 32

# print(env.observation_space.shape)
# print(nb_actions)
# assert False

actor = actor.build_actor(
    batch_size=batch_size,
    nb_actions=nb_actions,
    env=env,
    num_feat=num_feat)

critic = critic.build_critic(
    action_input=action_input,
    observation_input=observation_input,
    num_feat=num_feat)

# Optimizer
opt = Adam(lr=lr, clipnorm=1.0)
# opt = Adam(lr=lr, clipnorm=0.01)

# Build and compile agent
memory = SequentialMemory(
    limit=limit,
    window_length=batch_size)

random_process = OrnsteinUhlenbeckProcess(
    size=nb_actions,
    theta=.15,
    mu=0.,
    sigma=.3)

agent = DDPGAgent(
    nb_actions=nb_actions,
    actor=actor,
    critic=critic,
    critic_action_input=action_input,
    memory=memory,
    nb_steps_warmup_critic=num_steps_warmup_critic,
    nb_steps_warmup_actor=num_steps_warmup_actor,
    random_process=random_process,
    gamma=gamma,
    target_model_update=1e-2)

agent.compile(opt, metrics=['mae'])

history = agent.fit(
    env,
<<<<<<< HEAD
    nb_steps=nb_steps,
=======
    nb_steps=30000,
>>>>>>> 484bf4449eb0aada0173d026314e91633a6dcc8a
    visualize=True,
    verbose=0,
    nb_max_episode_steps=max_episode_steps)

<<<<<<< HEAD
print(history)

# After training is done, we save the final weights.
agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

=======
# # After training is done, we save the final weights.
agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
#%%
>>>>>>> 484bf4449eb0aada0173d026314e91633a6dcc8a

# Finally, evaluate our algorithm for 5 episodes.
agent.test(
    env,
    nb_episodes=5,
    visualize=True,
    nb_max_episode_steps=max_episode_steps)
