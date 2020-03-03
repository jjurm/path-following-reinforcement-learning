import gym
from gym import error, spaces, utils
from gym.utils import seeding
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
            1. change in forward u
            2. change in rotational w

        # Observation Space:
        #     1. forward u
        #     2. rotational w
        #     3. theta heading
        #     4. distance to next point

        Observation Space:
            1. forward u
            2. rotational w
            3. x
            4. y
            5. theta heading
        """
        self.seed(0)

        # Boundaries of the environment
        self.MIN_X, self.MAX_X = (-5, 5)
        self.MIN_Y, self.MAX_Y = (-5, 5)
        self.MIN_SPEED = -1
        self.MAX_SPEED = 1
        self.MIN_THETA = -np.pi
        self.MAX_THETA = np.pi

        # Environment parameters
        self.goal_pos = np.array([2, 2]) # goal position
        self.dist_threshold = 0.2 # how close to goal = reach goal
        self.dt = 0.1 # Time delta per step
        self.ticks = 0

        # Reward parameters
        self.alpha = 1 # How much to scale the reward

        # Action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([self.MIN_SPEED, self.MIN_SPEED], dtype=np.float32),
            high=np.array([self.MAX_SPEED, self.MAX_SPEED], dtype=np.float32),
            dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array([self.MIN_X, self.MIN_Y], dtype=np.float32),
            high=np.array([self.MAX_X, self.MAX_Y], dtype=np.float32),
            dtype=np.float32)

        # Visualisation variables
        self.viewer = None
        self.pathTrace = 50
        self.pathTraceSpace = 3
        self.pathTraceSpaceCounter = 0
        self.path = np.zeros([self.pathTrace,2])
        self.pathPtr = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_time(self):
        return self.ticks * self.dt

    def _get_new_state(self, u, w, x, y, theta, new_u, new_w):
        # Clip updates
        new_u = np.clip(new_u, self.MIN_SPEED, self.MAX_SPEED)
        new_w = np.clip(new_w, self.MIN_SPEED, self.MAX_SPEED)

        # Update and compute velocities
        avg_u = (u + new_u) / 2
        avg_w = (w + new_w) / 2

        theta_motion = theta + avg_w * self.dt / 2

        # Update state
        u = new_u
        w = new_w
        x += avg_u * np.cos(theta_motion) * self.dt
        y += avg_u * np.sin(theta_motion) * self.dt
        theta += avg_w * self.dt

        return np.array([u, w, x, y, theta])

    def _is_invalid(self):
        _, _, x, y, _ = self.state
        # Out of bounds
        return x < self.MIN_X or x > self.MAX_X or y < self.MIN_Y or y > self.MAX_Y

    def _is_goal_reached(self):
        return self._get_goal_dist() < self.dist_threshold

    def _is_done(self):
        return self._is_invalid() or self._is_goal_reached()

    def _get_dist(self, p1, p2):
        return np.linalg.norm((p1 - p2))

    def _get_goal_dist(self):
        return self._get_dist(self.state[[2, 3]], self.goal_pos)

    def _get_potential(self):
        """
        Checks how close to the robot is to the goal, and simply use reciprocal
        multiplied by some scaling factor.
        """
        bonus = 10

        potential = - self.alpha * self._get_goal_dist()
        if self._is_goal_reached():
            potential += bonus
        #reward += - np.log(np.abs(state[1]))

        return potential

    def _get_observation(self):
        _, _, x, y, theta = self.state

        goal_relative = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ]).dot(self.goal_pos - np.array([x, y]))

        #observation = np.array([
        #    np.math.atan2(goal_relative[1], goal_relative[0]),
        #    np.linalg.norm(goal_relative)
        #])

        return goal_relative

    def step(self, action: np.ndarray):
        """
        States are updated according to forward kinematics.

        x_dot = u · cos θ
        y_dot = u · sin θ
        θ_dot = ω

        x -> x_dot . dt
        y -> y_dot . dt
        theta -> theta_dot . dt

        Args:
            - action (tuple): changes to u and v.

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
        new_u, new_w = np.array(action) * (self.MAX_SPEED - self.MIN_SPEED) / 2
        u, w, x, y, theta = self.state
        prev_potential = self._get_potential()

        # Update variables
        self.state = self._get_new_state(u, w, x, y, theta, new_u, new_w)

        observation = self._get_observation()
        next_potential = self._get_potential()
        reward = next_potential - prev_potential
        done = self._is_done()
        info = {}

        self.ticks += 1

        # print('current_position: {}'.format((x, y)))
        curr_pos = np.array([self.state[2], self.state[3]])
        # print("Curr Dist: {:.4f}\t|\t Position: {}".format(
        #     self._get_dist(curr_pos, self.goal_pos),
        #     curr_pos))

        frequency = 2
        if done or self.ticks % np.round(1 / self.dt / frequency) == 0:
            print(f"T {self._get_time()}: Pos ({x}, {y}), action ({new_u}, {new_w}), Obs{observation}, reward {reward}")

        return observation, reward, done, info

    def reset(self):
        """
        Returns a random observation state.

        Called initially before an episode starts.
        """
        rand_u = np.random.uniform(self.MIN_SPEED, self.MAX_SPEED)
        rand_w = np.random.uniform(self.MIN_SPEED, self.MAX_SPEED)
        rand_x = np.random.uniform(self.MIN_X, self.MAX_X) / 5.
        rand_y = np.random.uniform(self.MIN_Y, self.MAX_Y) / 5.
        rand_theta = np.random.uniform(self.MIN_THETA, self.MAX_THETA)

        self.state = np.array([
            0,
            0,
            rand_x,
            rand_y,
            rand_theta])
        self.ticks = 0

        return self._get_observation()

    def render(self, mode='human'):
        if self.viewer is None:
            # import the required library
            from gym.envs.classic_control import rendering

            # Set the display window size and range
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(self.MIN_X,self.MAX_X,self.MIN_Y,self.MAX_Y) #Scale (X,X,Y,Y)

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

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

