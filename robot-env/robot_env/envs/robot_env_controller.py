import gym
import numpy as np
from gym import spaces
from simple_pid import PID

from .robot_env import RobotEnv


class RobotEnvController(RobotEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        super().__init__()
        self.pid = PID(1., 1., 1.,
                       sample_time=None)
        self.epsilon = 0.1
        self.get_virtual_position = self.get_holonomic_point

        n_params = 3
        self.action_space = spaces.Box(
            low=np.array([-np.inf] * n_params, dtype=np.float32),
            high=np.array([np.inf] * n_params, dtype=np.float32),
            dtype=np.float32)

    def step(self, action: np.ndarray):
        """
        :param action: [kp, ki, kd]
        :return: observation
        """
        # Set PID parameters
        self.pid.tunings = [0., 0., 0.]

        vector_forward = self.get_rough_direction()
        vector_perpendicular = self.get_cp_vector()
        sign = np.sign(np.cross(vector_forward, vector_perpendicular))
        vector_perpendicular_oriented = vector_perpendicular * sign

        # Use PID
        error_signed = np.linalg.norm(vector_perpendicular) * sign

        direction_forward = vector_forward / np.linalg.norm(vector_forward)
        direction_perpendicular = vector_perpendicular_oriented / np.abs(error_signed) if np.abs(error_signed) > 1e-7 else direction_forward

        desired_vector = -self.pid(error_signed) * direction_perpendicular + direction_forward
        #desired_vector = -direction_perpendicular

        direction_desired = desired_vector / np.linalg.norm(desired_vector)

        # Feedback linearisation
        theta = 0.
        u, w = np.array([
            [1., 0.],
            [0., 1. / self.epsilon]]
        ).dot(np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])).dot(
            direction_desired
        )

        # Step (observation, reward, done, info)
        return super().step(np.array([u, w]))

    def get_holonomic_point(self):
        """Returns absolute coordinates of the robot's holonomic point"""
        theta = self.sim.theta
        direction = np.array([np.cos(theta), np.sin(theta)])
        return self.sim.position + self.epsilon * direction
