"""Project template example running the environment.
"""
from __future__ import division
import time
from test_close_gripper_env import TestCloseGripperEnv
import logging
import numpy as np
logging.basicConfig(level=logging.DEBUG)


def get_action(observation):
    """Run your policy to produce an action.
    """
    action = [0.5, 0, 0]
    return action


env = TestCloseGripperEnv('projects/test_close_gripper/test_close_gripper.yaml', True, "TestCloseGripperEnv")
env.MAX_STEPS = 199

gripper_path = np.linspace(0.5, 0.1, num=10)
step = 0

#ac = [0.5, 0.9]

for ep_num in range(10):
    logging.debug('episode ' + str(ep_num - 1) + ' complete...pausing...')
    observation = env.reset()
    done = False
    
    while not done:
        start = time.time()
        action = get_action(observation)

        # Pass the start time to enforce policy frequency.

        if observation["new_comm_okay"]:
            #sprint ("New command")
            action[0] = 0.01 #gripper_path[step % 10] #ac[ep_num % 2]
            step += 1

            print (f"gripper_pos: {observation['gripper_pos']}")
            print (f"action: {action[0]}")
        else:
            #print ("old command")
            action[0] = env.gripper_des_val

        

        observation, reward, termination, info = env.step(action, start=start)

        done = termination
        
    step = 0

# In the real robot we have to use a ROS interface. Disconnect the interface
# after completing the experiment.
if (not env.world.is_sim):
    env.robot_interface.disconnect()
    env.sensor_interface.disconnect()
