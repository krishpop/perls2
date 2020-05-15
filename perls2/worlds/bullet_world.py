""" Parent class for Abstract Pybullet Environments
"""

import abc  # For abstract class definitions
import six  # For abstract class definitions
import pybullet
import numpy as np
import os
import time
import timeit
import logging

from perls2.worlds.world import World
from perls2.arenas.bullet_arena import BulletArena
from perls2.robots.bullet_robot_interface import BulletRobotInterface
from perls2.sensors.bullet_camera_interface import BulletCameraInterface
from perls2.objects.bullet_object_interface import BulletObjectInterface


class BulletWorld(World):
    """Class for PyBullet worlds.

    A BulletWorld will automatically create instances of BulletRobot,\
    BulletCamera and BulletObjectInterfaces based on the config file.

    Attributes:
        arena (BulletArena): Manages the sim by loading models, and for
        simulations, randomizing objects and sensors params.

        robot_interface (BulletRobotInterface): Communicates with robots and
            executes robot commands.

        sensor_interface (BulletSensorInterface): Retrieves sensor info and
            executes changes to params (e.g. intrinsics/extrinsics)

        object_interface (BulletObjectInterface): retrieves object info and
            executes changes to params

        physics_id (int): Unique id identifying physics client for Pybullet.
            Used to connect other interfaces when working with multiple
            simulations.

        time_step: float
            Float defining timestep to increment Pybullet simulation during
            pybullet.stepSimulation() calls. Set in config file

        MAX_STEPS: int
            Constant for maximum number of steps before terminating epsiode.
            Set in config file
        name: str
            A name to describe the world (e.g. training or testing)

    Methods:
        (These are similar to openAI.gym)
        step:
            Step env forward and return observation, reward, termination, info.
            Not typically user-defined but may be modified.
        reset:
            Reset env to initial setting and return observation.
            Some aspects such as randomization are user-defined
        render:
        close:
        seed:

    """

    def __init__(self,
                 config,
                 use_visualizer=False,
                 name=None):
        """Initialize.

        Args:
            config (dict): YAML config dict containing specifications.

            use_visualizer (bool): Whether or not to use visualizer.

                ..note::Pybullet only allows for one simulation to be
                connected to GUI. It is up to the user to manage this.

            name (str): Name of the world. (for multiple worlds)

        Returns:
            None

        Example:
            BulletWorld('cfg/my_config.yaml', True, 'MyBulletWorld')

        ..note::
            *The world_factory is the recommended way for creating an
            appropriate instance of world according to the config parameters.*
        ..note::
            Upon initialization the world automatically creates instances of
            BulletRobotInterface, BulletSensorInterface and
            BulletObjectInterface according to the config dictionary.

        """

        # Get configuration parameters
        self.config = config

        # Connect to appropriate pybullet channel based on use_visualizer flag
        self.use_visualizer = use_visualizer
        if self.use_visualizer:
            self._physics_id = pybullet.connect(pybullet.GUI)
        else:
            self._physics_id = pybullet.connect(pybullet.DIRECT)

        logging.info("New PhysicsID: " + str(self._physics_id))
        self._time_step = self.config['sim_params']['time_step']

        pybullet.setGravity(0, 0, -9.8, physicsClientId=self._physics_id)
        pybullet.setTimeStep(self._time_step, physicsClientId=self._physics_id)

        # Create an arena to load robot and objects
        self.arena = BulletArena(self.config, self._physics_id)

        self.robot_interface = BulletRobotInterface.create(
            config=self.config,
            physics_id=self._physics_id,
            arm_id=self.arena.arm_id, 
            controlType=self.config['controller']['selected_type'])
        
        self.control_freq = self.config['sim_params']['control_freq']

        self.sensor_interface = BulletCameraInterface(
            physics_id=self._physics_id,
            image_height=self.config['sensor']['camera']['image']['height'],
            image_width=self.config['sensor']['camera']['image']['width'], 
            cameraEyePosition=self.config['sensor']['camera']['extrinsics']['eye_position'],
            cameraTargetPosition=self.config['sensor']['camera']['extrinsics']['target_position'],
            cameraUpVector=self.config['sensor']['camera']['extrinsics']['up_vector']
            )

        # Create a dictionary of object interfaces
        self.objects = {}

        # Create object interfaces for each of the objects found in the arena
        # dictionary
        for obj_idx, obj_name in enumerate(self.arena.object_dict):
            self.objects[obj_name] = BulletObjectInterface(
                physics_id=self._physics_id,
                obj_id=self.arena.object_dict[obj_name],
                name=obj_name)
        # TODO: give world a method get_object_interface(str name)
        # TODO: make object default position a part of objectn not all objects.
        self.print_this_step = False

        self.name = name

        #To ensure smoothness of simulation and collisions, execute
        # a number of simulation steps per action received by policy
        self.ctrl_steps_per_action = self.config['sim_params']['steps_per_action']

        self.is_sim = True
        
        # TODO REMOVE DEBUGs    
        self.joint_num = 0
        self.dim_num = 0
        self.step_counter = 0

        self.ee_list = []

    @property
    def physics_id(self):
        return self._physics_id


    def add_object(self, path, name, pose, scale, is_static=False):
        """ Add object to world explicitly.

        Args:
            path (str): filepath name to object urdf
            name (str): name of object used for dictionary key
            pose (list): (7,) pose of the object as [x, y, z, qx, qy, qz, w]
                orientation as quaternion.
            scale (double): scale of object.
            is_static (bool): whether object should remain fixed or not.

        Returns:
            object_interface (ObjectInterface): object interface added to world.

        Examples:
            objI = self.world.add_object('objects/ycb/013_apple/google_16k/textured.urdf',
                                  '013_apple',
                                   [0, 0, 0, 0, 0, 0, 1],
                                   1.0,)
        """

        # Get the pybullet id from arena
        obj_id = self.arena._load_object_path(path, name, pose, scale, is_static=False)
        self.arena.object_dict[name] = obj_id

        # Create the BulletObject Interface
        object_interface = BulletObjectInterface(
                physics_id=self._physics_id,
                obj_id=obj_id,
                name=name)
        # Add to Objects dictionary
        self.objects[name] = object_interface

        return object_interface

    def remove_object(self, name):
        """ Remove object from world.
        Args:
            name (str): name of the object as stored in objects dictionary.

        Returns:
            -

        Notes: Removes object from arena as well as objects dictionary.
        """
        try:
            objectI = self.objects.pop(name)
            self.arena._remove_object(objectI.obj_id, objectI.physics_id)
        except:
            logging.ERROR('key not found')
        self.arena._remove_object(objectI.obj_id, objectI.physics_id)

    def reset(self):
        """Reset the world.

        Returns:
            None

        TODO: Should this return something? an error? This needs to be
        defined by derived class. But all derived reset functions share
        these steps.
        """
        pass

    def step(self):
        """Step the world(simulation) forward.

        Args:
            None
        Returns:
            None

        Step simulation forward a number of times per action
        to ensure smoothness when for collisions
        """
        # TODO: add real time option

        # Prepare for next step by executing action
        for exec_steps in range(self.ctrl_steps_per_action):
            self.run_control_loop_for_action()


        self.step_counter +=1
        #self.step_log = open('dev/logs/control/step' + str(self.step_counter) + '.txt', 'w+')

    def run_control_loop_for_action(self):
        for step in range(self.control_freq):
            # start = time.time()
            self.robot_interface.step()
            # print("robot_interface step(): " + str(time.time() - start))
            # start = time.time()
            pybullet.stepSimulation(self._physics_id)
            # print("pb step sim: " + str(time.time() - start))
    def get_observation(self):
        """Get observation of current env state

        Returns:
            An observation, typically a dict with multiple things

        This function is to be user-defined. Note that some tools require
        observations to be numpy arrays.
        """
        raise NotImplementedError

    def visualize(self, observation, action):
        """Visualize the action.

        Add visual markers to the world (in case of sim)
        or execute some movements (in case of real) to
        indicate the action about to be performed.

        Args:
            observation: The observation of the current step.
            action: The selected action.
        """
        pass

    def handle_exception(self, e):
        """Handle an exception.
        """
        pass

    @property
    def info(self):
        return {
                }

    def rewardFunction(self):
        """ User defined-Reward for agent given env state
        """
        raise NotImplementedError

    def check_stable(self,
                     linear_velocity_threshold,
                     angular_velocity_threshold):
        """Check if the loaded object is stable.

        Args:
            body: An instance of body.

        Returns:
            is_stable: True if the linear velocity and the angular velocity are
            almost zero; False otherwise.
        """
        linear_velocity = np.linalg.norm(body.linear_velocity)
        angular_velocity = np.linalg.norm(body.angular_velocity)

        if linear_velocity_threshold is None:
            has_linear_velocity = False
        else:
            has_linear_velocity = (
                    linear_velocity >= linear_velocity_threshold)

        if angular_velocity_threshold is None:
            has_angular_velocity = False
        else:
            has_angular_velocity = (
                    angular_velocity >= angular_velocity_threshold)

        is_stable = (not has_linear_velocity) and (not has_angular_velocity)

        return is_stable

    def _wait_until_stable(self,
                           linear_velocity_threshold=0.005,
                           angular_velocity_threshold=0.005,
                           check_after_steps=100,
                           min_stable_steps=100,
                           max_steps=30000):
        """Wait until the objects are stable.

        Blocking code that checks if object linear and angular velocity are
        within respectives thresholds of 0.

        Args:
            linear_velocity_threshold (float)
            angular_velocity_threshold (float)
            check_after_steps (int): min number of steps to wait before
                checking object state
            min_stable_step (int): min number of steps for object to be stable
                before exiting
            max_steps (int): max number of steps to wait for stable object

        Returns:
            None

        :TODO:
            * add check for angular velocity threshold
        """
        assert self.is_sim, 'This function is only used in simulation.'

        # logger.debug('Waiting for objects to be stable...')

        num_steps = 0
        num_stable_steps = 0

        while(1):
            pybullet.stepSimulation(self._physics_id)
            num_steps += 1
            all_stable = True

            if num_steps < check_after_steps:
                continue

            for obj_idx, obj_key in enumerate(self.objects):
                if (np.linalg.norm(self.objects[obj_key].linear_velocity) >=
                        linear_velocity_threshold):
                    all_stable = False

                if all_stable:
                    num_stable_steps +=1

            if ((num_stable_steps >= min_stable_steps) or
                    (num_steps >= max_steps)):

                break
