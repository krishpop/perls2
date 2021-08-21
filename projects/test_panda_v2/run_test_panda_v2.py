"""Project template example running the environment.
"""
from __future__ import division
import time
from test_panda_v2_env import TestPandaEnv
from demos.demo_path import Line
import perls2.controllers.utils.transform_utils as T
import numpy as np

import logging
logging.basicConfig(level=logging.DEBUG)

DELTA = 0.002
TOTAL_STEPS = 30

def get_goal_poses(initial_pose, step_ct, delta_val, axis=0):
    path = Line(start_pose=initial_pose, 
                num_pts=step_ct, 
                delta_val=delta_val,
                dim=axis)
    return path.path

def get_delta(goal_pose, current_pose):
        """Get delta between goal pose and current_pose.

        Args: goal_pose (list): 7f pose [x, y, z, qx, qy, qz, w] . Position and quaternion of goal pose.
              current_pose (list): 7f Position and quaternion of current pose.

        Returns: delta (list) 6f [dx, dy, dz, ax, ay, az] delta position and delta orientation
            as axis-angle.
        """

        if len(goal_pose) != 7:
            raise ValueError("Goal pose incorrect dimension should be 7f")
        if len(current_pose) !=7:
            raise ValueError("Current pose incorrect dimension, should be 7f")

        dpos = np.subtract(goal_pose[:3], current_pose[:3])
        goal_mat = T.quat2mat(goal_pose[3:])
        current_mat = T.quat2mat(current_pose[3:])
        delta_mat_T = np.dot(goal_mat, current_mat.T)
        delta_quat = T.mat2quat(np.transpose(delta_mat_T))
        delta_aa = T.quat2axisangle(delta_quat)

        return np.hstack((dpos, delta_quat)).tolist()
        #return np.hstack((dpos, delta_aa)).tolist()

def get_action(observation, goal_pose):
    """
    Returns the required motion delta to move the end-effector from the current to the goal position
    Args: observation (dict) contains {"ee_pose": 7f end-effector pose, "q": 7f joint pose}
    Returns: delta (list) 6f [dx, dy, dz, ax, ay, az] delta position and delta orientation
            as axis-angle.
    """

    ee_pose = observation["ee_pose"]
    pose_delta = get_delta(goal_pose, ee_pose)
    return pose_delta


env = TestPandaEnv('projects/test_panda_v2/test_panda_v2.yaml', True, "TestPandaEnv")
env.MAX_STEPS = TOTAL_STEPS

for ep_num in range(10):
    logging.debug('episode ' + str(ep_num - 1) + ' complete...pausing...')
    step = 0
    observation = env.reset()
    path_idx = 0

    initial_pose = observation["ee_pose"]
    goal_poses = get_goal_poses(initial_pose, TOTAL_STEPS, DELTA, axis=0)

    done = False

    while not done:
        start = time.time()
        
        goal_pose = goal_poses[-1]#goal_poses[path_idx]
        action = get_action(observation, goal_pose)

        # Pass the start time to enforce policy frequency.
        observation, reward, termination, info = env.step(action, start=start)

        path_idx += 1
        done = termination

# In the real robot we have to use a ROS interface. Disconnect the interface
# after completing the experiment.
if (not env.world.is_sim):
    env.robot_interface.disconnect()
    env.sensor_interface.disconnect()
