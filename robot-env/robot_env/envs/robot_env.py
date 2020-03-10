import gym
import numpy as np
from gym import spaces

from .simulation import Simulation
from .viewer import Viewer


class RobotEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        """
        Action spaces
            1. forward u
            2. change in rotational w

        Observation Space:
            1. forward u
            2. rotational w
            3. x
            4. y
            5. theta heading
        """
        self.dt = 0.1  # Time delta per step
        self.sim = Simulation(self.dt)
        self.seed(0)

        # Boundaries of the environment
        self.MIN_SPEED = 0.2
        self.MAX_SPEED = 1.
        self.MAX_THETA = 2.

        # Environment parameters
        self.goal_pos = np.array([3, 0])  # goal position
        self.dist_threshold = 0.5  # how close to goal = reach goal

        # Action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([self.MIN_SPEED, -self.MAX_THETA], dtype=np.float32),
            high=np.array([self.MAX_SPEED, self.MAX_THETA], dtype=np.float32),
            dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array([-self.sim.FIELD_SIZE, -self.sim.FIELD_SIZE], dtype=np.float32),
            high=np.array([self.sim.FIELD_SIZE, self.sim.FIELD_SIZE], dtype=np.float32),
            dtype=np.float32)

        self.testItr = 0
        self.relative_goal_pos = self.goal_pos
        self.debug = [0, 0, 0, 0, 0, 0, 0, 0]

        # Visualisation variables
        self.viewer = None

    def _is_goal_reached(self):
        """
        Check if goal is reached.
        """
        return self._get_goal_dist() < self.dist_threshold

    def _is_done(self):
        return self.sim.is_invalid() or self._is_goal_reached()

    def _get_dist(self, p1: np.ndarray, p2: np.ndarray):
        return np.linalg.norm(p1 - p2)

    def _get_goal_dist(self):
        return self._get_dist(self.sim.position, self.goal_pos)

    def _get_reward(self):
        u, w = self.sim.speed
        x, y = self.sim.position
        theta = self.sim.theta

        reward_distance = 0

        next_x, next_y = (x + u * np.cos(theta) * self.sim.dt, y + u * np.cos(theta) * self.dt)
        next_pos = np.array([next_x, next_y])
        if self._get_goal_dist() < 2.:
            reward_distance += (2.1 - self._get_dist(next_pos, self.goal_pos)) ** 4 * np.cos(
                np.clip(self.sim.theta * 2, -np.pi, np.pi))
        else:
            reward_distance = self._get_goal_dist() - self._get_dist(next_pos, self.goal_pos)

        reward_directional = (np.pi - np.abs(self.sim.theta) * 5) * 0.1
        if reward_directional < 0:
            reward_directional *= 4
            if reward_directional < -np.pi * 2:
                reward_directional = -np.pi * 2

        reward = reward_distance + reward_directional - np.abs(w) * 0.1

        return reward

    def _get_observation(self):
        x, y = self.sim.position
        theta = self.sim.theta

        goal_relative = np.array([
            [np.cos(-theta), -np.sin(-theta)],
            [np.sin(-theta), np.cos(-theta)]
        ]).dot(self.goal_pos - np.array([x, y]))

        return goal_relative

    def step(self, action: np.ndarray):
        """
        Args:
            - action (tuple): u and change in v.
        Returns:
            - observation (object):
            - reward (float):
            - done (boolean):
            - info (dict):
        """

        u, w = action
        u = (np.tanh(u) + 1) / 2 * (
                self.MAX_SPEED - self.MIN_SPEED) + self.MIN_SPEED  # normalize the range of action is -1.5 to 1.5
        w = np.tanh(w) * self.MAX_THETA  # normalize

        self.sim.step(np.array([u, w]))

        return (self._get_observation()), (self._get_reward()), self._is_done(), {}

    def _print_info(self, reward):
        frequency = 50
        if self._is_done() or self.sim.ticks % np.round(1 / self.dt / frequency) == 0:
            u, w = self.sim.speed
            x, y = self.sim.position

            print(f"T {self.sim.time}: Pos ({x:.4f}, {y:.4f}), action ({u:.4f}, {w:.4f}), reward {reward}")

    def reset(self):
        self.sim.reset()
        return self._get_observation()

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = Viewer(self)
        self.viewer.render(mode)

    def close(self):
        if self.viewer:
            self.viewer.viewer.close()
            self.viewer = None
