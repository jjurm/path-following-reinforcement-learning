import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

class RobotEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Action spaces:
            1. change in forward u
            2. change in rotational w

        Observation Space:
            1. forward u
            2. rotational w
            3. heading
            4. distance to next point
        """
        self.MIN_SPEED = -3
        self.MAX_SPEED = 3
        self.MAX_DIST = 15

        self.action_space = spaces.Tuple(
            spaces.Box(
                low=self.MIN_SPEED,
                high=self.MAX_SPEED,
                shape=1),
            spaces.Box(
                low=self.MIN_SPEED,
                high=self.MAX_SPEED,
                shape=1))

        self.observation_space = spaces.Tuple(
            spaces.Box(
                low=self.MIN_SPEED,
                high=self.MAX_SPEED,
                shape=1),
            spaces.Box(
                low=self.MIN_SPEED,
                high=self.MAX_SPEED,
                shape=1),
            spaces.Box(
                low=-np.pi,
                high=np.pi,
                shape=1),
            spaces.Box(
                low=0,
                high=self.MAX_DIST))

        # Boundaries of the BOX
        self.X_MIN, self.X_MAX = (-5, 5)
        self.Y_MIN, self.Y_MAX = (-5, 5)

        # Goal position and threshold
        self.goal_pos = np.array((4, 4))
        self.dist_threshold = 0.2 # how close to goal = reach goal

        # Time delta
        self.dt = 0.1

        # History of paths taken by robot
        self.path_history = {}

    def step(self, action):
        """
        Returns 4 values:

        - observation (object): 
            an environment-specific object representing your observation of the environment. For example, pixel data from a camera, joint angles and joint velocities of a robot, or the board state in a board game.
        - reward (float): 
            amount of reward achieved by the previous action. The scale varies between environments, but the goal is always to increase your total reward.
        - done (boolean): 
            whether it’s time to reset the environment again. Most (but not all) tasks are divided up into well-defined episodes, and done being True indicates the episode has terminated. (For example, perhaps the pole tipped too far, or you lost your last life.)
        - info (dict):
            diagnostic information useful for debugging. It can sometimes be useful for learning (for example, it might contain the raw probabilities behind the environment’s last state change). However, official evaluations of your agent are not allowed to use this for learning.
        """
        # Unpack deltas
        delta_u, delta_w = action.spaces

        if not self.action_space.contains(action):
            raise ValueError("Invalid action {}".format(action))

        # Compute new state values
        new_u = 

        self.state = 




    def reset(self):
        """
        Returns a random observation state.

        Called initially before an episode starts.
        """
        rand_u = np.random.uniform(self.MIN_SPEED, self.MAX_SPEED)
        rand_w = np.random.uniform(self.MIN_SPEED, self.MAX_SPEED)
        rand_theta = np.random.uniform(-np.pi, np.pi)
        rand_dist = np.random.uniform(0, self.MAX_DIST)

        self.state = np.array([
            rand_u,
            rand_w,
            rand_theta,
            rand_dist])

        return self.state       

    def render(self, mode='human'):
        pass

    def close(self):
        pass
