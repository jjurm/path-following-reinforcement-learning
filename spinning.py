# from spinup.utils.run_utils import ExperimentGrid
# from spinup import ppo_pytorch
# import torch

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--cpu', type=int, default=4)
#     parser.add_argument('--num_runs', type=int, default=3)
#     args = parser.parse_args()

#     eg = ExperimentGrid(name='ppo-pyt-bench')
#     eg.add('env_name', 'CartPole-v0', '', True)
#     eg.add('seed', [10*i for i in range(args.num_runs)])
#     eg.add('epochs', 10)
#     eg.add('steps_per_epoch', 4000)
#     eg.add('ac_kwargs:hidden_sizes', [(32,), (64,64)], 'hid')
#     eg.add('ac_kwargs:activation', [torch.nn.Tanh, torch.nn.ReLU], '')
#     eg.run(ppo_pytorch, num_cpu=args.cpu)


#========

from spinup.utils.run_utils import ExperimentGrid
<<<<<<< HEAD
from spinup import ppo_pytorch, ddpg_pytorch, sac_pytorch
=======
from spinup import ppo_pytorch
>>>>>>> 484bf4449eb0aada0173d026314e91633a6dcc8a
import torch
import gym

if __name__ == '__main__':
    ############
    #   Setup
    ############
    ENV_NAME = 'robot_env:robot-env-v0'
    env = gym.make(ENV_NAME)
    feat = 32


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=3)
    args = parser.parse_args()

    eg = ExperimentGrid(name='ppo-pyt-bench')
    eg.add('env_name', ENV_NAME, '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 20)
    eg.add('steps_per_epoch', 4000)
    # eg.add('ac_kwargs:hidden_sizes', [(feat,), (feat*2,feat*2)], 'hid')
    # eg.add('ac_kwargs:activation', [torch.nn.Tanh, torch.nn.ReLU], '')
    eg.add('ac_kwargs:hidden_sizes', [(feat*2,feat*2)], 'hid')
    eg.add('ac_kwargs:activation', [torch.nn.ReLU], '')

<<<<<<< HEAD
    # eg.run(ppo_pytorch, num_cpu=args.cpu)
    eg.run(ddpg_pytorch, num_cpu=args.cpu)
=======
    eg.run(ppo_pytorch, num_cpu=args.cpu)
>>>>>>> 484bf4449eb0aada0173d026314e91633a6dcc8a
