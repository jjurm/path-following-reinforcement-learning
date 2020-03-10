import gym
from gym import error, spaces, utils
from os import path
import numpy as np

class RobotEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
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
        self.seed(0)

        # Boundaries of the environment
        self.FIELD_SIZE = 5.
        self.MIN_SPEED = 0.2
        self.MAX_SPEED = 1.
        self.MAX_THETA = 2.

        # Environment parameters
        self.goal_pos = np.array([3, 0]) # goal position
        self.dist_threshold = 0.5 # how close to goal = reach goal
        self.dt = 0.1 # Time delta per step
        self.ticks = 0

        # Action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([self.MIN_SPEED, -self.MAX_THETA], dtype=np.float32),
            high=np.array([self.MAX_SPEED, self.MAX_THETA], dtype=np.float32),
            dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array([-self.FIELD_SIZE, -self.FIELD_SIZE], dtype=np.float32),
            high=np.array([self.FIELD_SIZE, self.FIELD_SIZE], dtype=np.float32),
            dtype=np.float32)

        self.testItr = 0
        self.relative_goal_pos = self.goal_pos
        self.debug = [0,0,0,0,0,0,0,0]

        # Visualisation variables
        self.viewer = None
        self.pathTrace = 50
        self.pathTraceSpace = 3
        self.pathTraceSpaceCounter = 0
        self.path = np.zeros([self.pathTrace,2])
        self.pathPtr = 0

    def _get_time(self):
        return round(self.ticks * self.dt, 4)

    def _is_invalid(self):
        """
        Check if out of bounds.
        """
        _, _, x, y, _ = self.state
        return x < -self.FIELD_SIZE or x > self.FIELD_SIZE or y < -self.FIELD_SIZE or y > self.FIELD_SIZE

    def _is_goal_reached(self):
        """
        Check if goal is reached.
        """
        return self._get_goal_dist() < self.dist_threshold

    def _is_done(self):
        return self._is_invalid() or self._is_goal_reached()

    def _get_dist(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def _get_goal_dist(self):
        return self._get_dist(self.state[[2, 3]], self.goal_pos)

    def _get_reward(self):
        u, w, x, y, theta = self.state

        reward_distance = 0
        reward_directional = 0

        next_x, next_y = (x + u * np.cos(theta) * self.dt, y + u * np.cos(theta) * self.dt)
        if self._get_goal_dist() < 2.:
            reward_distance += (2.1 - self._get_dist([next_x, next_y], self.goal_pos)) ** 4 * np.cos(np.clip(self.state[4] * 2,-np.pi,np.pi))
        else:
            reward_distance = self._get_goal_dist() - self._get_dist([next_x, next_y], self.goal_pos)

        reward_directional = (np.pi - np.abs(self.state[4]) * 5) * 0.1
        if reward_directional < 0:
            reward_directional *= 4
            if reward_directional < -np.pi * 2:
                reward_directional = -np.pi * 2

        return reward_distance + reward_directional - np.abs(w) * 0.1

    def _get_new_state(self, next_u, next_w):
        next_u = (np.tanh(next_u) + 1) / 2 * (self.MAX_SPEED - self.MIN_SPEED) + self.MIN_SPEED # normalize the range of action is -1.5 to 1.5
        next_w = np.tanh(next_w) * self.MAX_THETA # normalize

        u, w, x, y, theta = self.state

        # Update state
        new_theta = theta + w * self.dt
        while new_theta > np.pi:
            new_theta -= np.pi * 2
        while new_theta < -np.pi:
            new_theta += np.pi * 2
        new_x = x + u * np.cos(theta) * self.dt
        new_y = y + u * np.sin(theta) * self.dt

        return np.array([next_u, next_w, new_x, new_y, new_theta])

    def _get_observation(self):
        _, _, x, y, theta = self.state

        goal_relative = np.array([
            [np.cos(-theta), -np.sin(-theta)],
            [np.sin(-theta), np.cos(-theta)]
        ]).dot(self.goal_pos - np.array([x, y]))

        return goal_relative

    def step(self, action: np.ndarray):
        """
        States are updated according to forward kinematics.

        Args:
            - action (tuple): u and change in v.

        Returns:
            - observation (object):
                an environment-specific object representing your observation of the environment.
                For example, pixel data from a camera, joint angles and joint velocities of a robot, or the board state in a board game.
            - reward (float):
                amount of reward achieved by the previous action.
                The scale varies between environments, but the goal is always to increase your total reward.
            - done (boolean):
                whether it’s time to reset the environment again.
                Most (but not all) tasks are divided up into well-defined episodes, and done being True indicates the episode has terminated.
                (For example, perhaps the pole tipped too far, or you lost your last life.)
            - info (dict):
                diagnostic information useful for debugging.
                It can sometimes be useful for learning (for example, it might contain the raw probabilities behind the environment’s last state change).
                However, official evaluations of your agent are not allowed to use this for learning.
        """
        # Unpack current state and action to take
        u, w = action

        # Update variables
        self.state = self._get_new_state(u, w)

        observation = self._get_observation()
        reward = self._get_reward()
        done = self._is_done()
        info = {}
        self.ticks += 1

        frequency = 50
        if done or self.ticks % np.round(1 / self.dt / frequency) == 0:
            if self._is_invalid():
                reward -= 100
            if self._is_goal_reached():
                reward += 50
                reward += 25 / self._get_time()
            else:
                reward -= 10
            u, w, x, y, theta = self.state
            print(f"T {self._get_time()}: Pos ({x:.4f}, {y:.4f}), action ({u:.4f}, {w:.4f}), reward {reward}")

        return observation, reward, done, info

    def reset(self):
        """
        Returns a random observation state.

        Called initially before an episode starts.
        """
        rand_x = np.random.uniform(-self.FIELD_SIZE, self.FIELD_SIZE) / 5.
        rand_y = np.random.uniform(-self.FIELD_SIZE, self.FIELD_SIZE) / 5.
        rand_theta = np.random.uniform(-np.pi, np.pi)

        self.testItr += 1
        self.state = np.array([
            0,
            0,
            0,
            0,
            0])
        if self.testItr > 25:
            self.state = np.array([
                0,
                0,
                rand_x,
                rand_y,
                0])
            while self._get_dist(self.state[[2,3]], self.goal_pos) < 0.5:
                rand_x = np.random.uniform(-self.FIELD_SIZE, self.FIELD_SIZE) / 5.
                rand_y = np.random.uniform(-self.FIELD_SIZE, self.FIELD_SIZE) / 5.
                self.state = np.array([
                    0,
                    0,
                    rand_x,
                    rand_y,
                    0])
        if self.testItr > 50:
            self.state = np.array([
                0,
                0,
                rand_x * 3,
                rand_y * 3,
                rand_theta])
            while self._get_dist(self.state[[2,3]], self.goal_pos) < 0.5:
                rand_x = np.random.uniform(-self.FIELD_SIZE, self.FIELD_SIZE) / 5.
                rand_y = np.random.uniform(-self.FIELD_SIZE, self.FIELD_SIZE) / 5.
                self.state = np.array([
                    0,
                    0,
                    rand_x * 3,
                    rand_y * 3,
                    rand_theta])
        self.ticks = 0
        if self.testItr > 80 and self.testItr % 10 == 0:
            self.goal_pos = np.array([np.random.uniform(-self.FIELD_SIZE, self.FIELD_SIZE)*0.9, np.random.uniform(-self.FIELD_SIZE, self.FIELD_SIZE)*0.9])
            while self._get_dist(self.state[[2,3]], self.goal_pos) < 0.5:
                rand_x = np.random.uniform(-self.FIELD_SIZE, self.FIELD_SIZE) / 5.
                rand_y = np.random.uniform(-self.FIELD_SIZE, self.FIELD_SIZE) / 5.
                self.state = np.array([
                    0,
                    0,
                    rand_x * 3,
                    rand_y * 3,
                    rand_theta])

        if self.testItr == 100:
            self.dist_threshold = 0.3
        return self._get_observation()

    def render(self, mode='human'):
        if self.viewer is None:
            # import the required library
            from gym.envs.classic_control import rendering

            # Set the display window size and range
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-self.FIELD_SIZE,self.FIELD_SIZE,-self.FIELD_SIZE,self.FIELD_SIZE) #Scale (X,X,Y,Y)

            # Create the robot
            fname = path.join(path.dirname(__file__), "assets/robot.png")
            self.robotobj = rendering.Image(fname, .25, .25)
            self.robot_t = rendering.Transform()
            self.robotobj.add_attr(self.robot_t)

            # Create the goal location
            self.goalobj = rendering.make_circle(.1)
            self.goalobj.set_color(255,0,0)
            self.goal_t = rendering.Transform()
            self.goalobj.add_attr(self.goal_t)
            self.viewer.add_geom(self.goalobj)
            self.goal_t.set_translation(self.goal_pos[0],self.goal_pos[1])

            # Create trace path
            self.traceobj = []
            self.traceobj_t = []
            for i in range(self.pathTrace):
                self.traceobj.append(rendering.make_circle(.02+.03 * i/self.pathTrace))
                print(.5*i/self.pathTrace,1.-.5*i/self.pathTrace,i/self.pathTrace)
                self.traceobj[i].set_color(.5-.5*i/self.pathTrace,1.-.5*i/self.pathTrace,i/self.pathTrace) # Setting the color gradiant for path
                self.traceobj_t.append(rendering.Transform())
                self.traceobj[i].add_attr(self.traceobj_t[i])
                self.traceobj_t[i].set_translation(-2+i*0.05,0)
                self.viewer.add_geom(self.traceobj[i])

        # Draw the robot
        self.viewer.add_onetime(self.robotobj)
        self.robot_t.set_translation(self.state[2],self.state[3])
        self.robot_t.set_rotation(self.state[4]-np.pi/2)

        # Update trace
        self.pathTraceSpaceCounter = (self.pathTraceSpaceCounter+1) % self.pathTraceSpace
        if self.pathTraceSpaceCounter == 0:
            self.path[self.pathPtr][0],self.path[self.pathPtr][1] = (self.state[2],self.state[3])
            self.pathPtr = (self.pathPtr+1) % self.pathTrace
            for i in range(self.pathTrace):
                counter = (i + self.pathPtr) % self.pathTrace
                self.traceobj_t[i].set_translation(self.path[counter][0],self.path[counter][1])

        self.goal_t.set_translation(self.goal_pos[0],self.goal_pos[1])
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
