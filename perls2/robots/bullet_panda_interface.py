"""
Class defining the interface to the Panda Robot in Bullet
"""

import pybullet
import numpy as np
import rbdl 
from perls2.robots.bullet_robot_interface import BulletRobotInterface


class BulletPandaInterface(BulletRobotInterface):
    """ Class for Panda Robot Interface in Pybullet. This class provides the
    functions for information about the state of the robot as well as sending
    commands.

    Attributes:
        physics_id (int): unique identifer for pybullet sim.
        arm_id (int) : unique identifier produced by pybullet to id robot.
        config (dict) : dictionary with configuration params for robot
        controlType (str): id for controlType ('osc', 'joint_space')
        limb_neutral_positiosn (list): list of joint angles for default

    """
    def __init__(self,
                 physics_id,
                 arm_id,
                 config=None,
                 controlType='EEImp'):

        self.data_dir = config['data_dir']
        import os
        model_path = os.path.join(self.data_dir, 'robot/franka_panda/panda.urdf')
        self.rbdl_model = rbdl.loadModel(bytes(model_path, 'utf-8'))
        super().__init__(physics_id, arm_id, config, controlType)

        self._ee_index = self.get_link_id_from_name('panda_link7')

        # Neutral positions
        self.limb_neutral_positions =  self.robot_cfg['neutral_joint_angles']

        self._name = "Franka Panda"
        self._default_force = 100
        self._default_position_gain = 0.1
        self._default_velocity_gain = 2.5
        #self.num_joints = 9

    def version(self):
        """dict of current versions of robot SDK, gripper, and robot
        """
        raise NotImplementedError

    def disconnect(self):
        pass