import numpy as np


class Simulation:

    def __init__(self, dt):
        self.FIELD_SIZE = 5.
        self._state = np.array([0., 0., 0., 0., 0.])
        self.ticks = 0
        self.dt = dt

    @property
    def time(self):
        return round(self.ticks * self.dt, 4)

    def is_invalid(self):
        """
        Check if out of bounds.
        """
        _, _, x, y, _ = self._state
        return x < -self.FIELD_SIZE or x > self.FIELD_SIZE or y < -self.FIELD_SIZE or y > self.FIELD_SIZE

    @property
    def speed(self):
        return self._state[[0, 1]]

    @property
    def position(self):
        return self._state[[2, 3]]

    @property
    def theta(self):
        return self._state[4]

    def step(self, action):
        next_u, next_w = action

        u, w, x, y, theta = self._state

        # Update state
        new_theta = theta + w * self.dt
        while new_theta > np.pi:
            new_theta -= np.pi * 2
        while new_theta < -np.pi:
            new_theta += np.pi * 2
        new_x = x + u * np.cos(theta) * self.dt
        new_y = y + u * np.sin(theta) * self.dt

        self.ticks += 1

        self._state = np.array([next_u, next_w, new_x, new_y, new_theta])

    def reset(self):
        rand_x = np.random.uniform(-self.FIELD_SIZE, self.FIELD_SIZE) / 5.
        rand_y = np.random.uniform(-self.FIELD_SIZE, self.FIELD_SIZE) / 5.
        rand_theta = np.random.uniform(-np.pi, np.pi)

        self._state = np.array([
            0,
            0,
            rand_x,
            rand_y,
            rand_theta
        ])
        self.start_state = self._state.copy()
