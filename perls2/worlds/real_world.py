""" Parent class for Mujoco Pybullet Environments
"""

import abc
import time
from perls2.worlds.world import World
from perls2.arenas.real_arena import RealArena
from perls2.robots.real_robot_interface import RealRobotInterface
from perls2.sensors.kinect_camera_interface import KinectCameraInterface


class RealWorld(World):
    """
    Typically, an env will have one robot with one camera as well as
    logic to have the robot execute a particular task.
    """

    def __init__(self,
                 config=None,
                 use_visualizer=False,
                 name='DefaultEnv'):
        """ Initialize.

        Parameters
        ----------
        config: dict
            A dict with config parameters. For initialization the relevant
            parameters that should be defined are:
             - robot:type
             - sensor:type
        """
        self.config = config

        self.name = name
        self.use_visualizer = use_visualizer

        # Learning parameters
        self.episode_num = 0

        self.arena = RealArena(self.config)

        self.robot_interface = RealRobotInterface.create(
            config=self.config,
            controlType=self.config['world']['controlType'])

        
        self.has_camera = False
        if 'sensor' in self.config:
            if 'real_camera' in self.config['sensor']:
                self.has_camera = True
                # self.camera_interface = KinectCameraInterface(self.config)


        self.is_sim = False

        self.dim_num = 0
        self.has_object = False
        self.last_step_call = None

    def reset(self):
        """Reset the environment.

        Returns:
            The observation.
        """
        # reload robot to restore body after any collisions
        #self.robot_interface.reset()
        pass 
        
    def step(self, start=None):
        """Take a step.

        Args:
            start (float): time.time() timestamp taken from before policy computes action.
                This is to enforce policy frequency. If start is None, policy frequency is 
                not enforced.

        Returns: None

        """
        self.robot_interface.step()

        if start is not None:
            while (time.time() - start) < (1./float(self.config['policy_freq'])):
                pass
        self.action_set = False
