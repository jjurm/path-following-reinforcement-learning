import numpy as np
from os import path
from gym.envs.classic_control import rendering

class Viewer:
    def __init__(self, env):
        self.env = env
        self.sim = env.sim

        self.pathTrace = 50
        self.pathTraceSpace = 3
        self.pathTraceSpaceCounter = 0
        self.path = np.zeros([self.pathTrace, 2])
        self.pathPtr = 0

        # Set the display window size and range
        self.viewer = rendering.Viewer(500, 500)
        self.viewer.set_bounds(-self.sim.FIELD_SIZE, self.sim.FIELD_SIZE, -self.sim.FIELD_SIZE,
                               self.sim.FIELD_SIZE)  # Scale (X,X,Y,Y)

        # Create the robot
        fname = path.join(path.dirname(__file__), "assets/robot.png")
        self.robotobj = rendering.Image(fname, .25, .25)
        self.robot_t = rendering.Transform()
        self.robotobj.add_attr(self.robot_t)

        # Create the goal location
        self.goalobj = rendering.make_circle(.1)
        self.goalobj.set_color(255, 0, 0)
        self.goal_t = rendering.Transform()
        self.goalobj.add_attr(self.goal_t)
        self.viewer.add_geom(self.goalobj)
        self.goal_t.set_translation(*self.env.goal_pos)

        # Create trace path
        self.traceobj = []
        self.traceobj_t = []
        for i in range(self.pathTrace):
            self.traceobj.append(rendering.make_circle(.02 + .03 * i / self.pathTrace))
            print(.5 * i / self.pathTrace, 1. - .5 * i / self.pathTrace, i / self.pathTrace)
            self.traceobj[i].set_color(.5 - .5 * i / self.pathTrace, 1. - .5 * i / self.pathTrace,
                                       i / self.pathTrace)  # Setting the color gradiant for path
            self.traceobj_t.append(rendering.Transform())
            self.traceobj[i].add_attr(self.traceobj_t[i])
            self.traceobj_t[i].set_translation(-2 + i * 0.05, 0)
            self.viewer.add_geom(self.traceobj[i])

    def render(self, mode='human'):
        # Draw the robot
        self.viewer.add_onetime(self.robotobj)
        self.robot_t.set_translation(self.sim.position[0], self.sim.position[1])
        self.robot_t.set_rotation(self.sim.theta - np.pi / 2)

        # Update trace
        self.pathTraceSpaceCounter = (self.pathTraceSpaceCounter + 1) % self.pathTraceSpace
        if self.pathTraceSpaceCounter == 0:
            self.path[self.pathPtr][0], self.path[self.pathPtr][1] = (self.sim.position[0], self.sim.position[1])
            self.pathPtr = (self.pathPtr + 1) % self.pathTrace
            for i in range(self.pathTrace):
                counter = (i + self.pathPtr) % self.pathTrace
                self.traceobj_t[i].set_translation(self.path[counter][0], self.path[counter][1])

        self.goal_t.set_translation(*self.env.goal_pos)
        output = self.viewer.render(return_rgb_array=mode == 'rgb_array')

        return output
