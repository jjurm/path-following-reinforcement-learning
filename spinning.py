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
from spinup import ppo_pytorch, ddpg_pytorch, sac_pytorch
import torch
import gym

if __name__ == '__main__':
    ############
    #   Setup
    ############
    ENV_NAME = 'robot_env:robot-env-v0'
    #ENV_NAME = 'robot_env:robot-env-controller-v0'
    env = gym.make(ENV_NAME)
    feat = 32

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--algo', required=True)
    parser.add_argument('--name', default=None)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    if args.name is None:
        args.name = args.algo

    eg = ExperimentGrid(name=args.name)
    eg.add('env_name', ENV_NAME, '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', args.epochs)
    eg.add('steps_per_epoch', 4000)
    # eg.add('ac_kwargs:hidden_sizes', [(feat,), (feat*2,feat*2)], 'hid')
    # eg.add('ac_kwargs:activation', [torch.nn.Tanh, torch.nn.ReLU], '')
    eg.add('ac_kwargs:hidden_sizes', [(feat*2,feat*2)], 'hid')
    eg.add('ac_kwargs:activation', [torch.nn.ReLU], '')

    if args.algo == 'ppo':
        eg.run(ppo_pytorch, num_cpu=args.cpu)

    elif args.algo == 'ddpg':
        eg.run(ddpg_pytorch, num_cpu=args.cpu)

    elif args.algo == 'sac':
        eg.run(sac_pytorch, num_cpu=args.cpu)

    else:
        raise ValueError("invalid algo.")
