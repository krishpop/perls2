"""Project template example running the environment.
"""
from __future__ import division
import time
from test_gripper_limit_env import TestGripperLimitEnv
import logging
logging.basicConfig(level=logging.DEBUG)



def get_action(observation):
    """Run your policy to produce an action.
    """
    action = [0, 0, 0]
    return action


env = TestGripperLimitEnv('projects/test_gripper_limit/test_gripper_limit.yaml', True, "TestGripperLimitEnv")
env.MAX_STEPS = 30

lower_limit = None
upper_limit = None

print ("================Gripper Tester===============")
print("type 'l' to set the lower gripper limit")
print("type 'u' to set the upper gripper limit")
print("type 'e' to exit")

for ep_num in range(1):
    logging.debug('episode ' + str(ep_num - 1) + ' complete...pausing...')
    step = 0
    observation = env.reset()
    done = False

    while not done:
        start = time.time()
        action = get_action(observation)

        # Pass the start time to enforce policy frequency.
        inp = str(input("Press Enter to Print Gripper Position: "))
        
        observation, reward, termination, info = env.step(action, start=start)
        pos = observation["gripper_position"]

        if inp == 'l':
            print (f"lower limit set to: {pos}")
            lower_limit = pos
        elif inp == 'u':
            if lower_limit is not None and lower_limit > pos:
                continue
            
            print(f"upper limit set to: {pos}")
            upper_limit = pos
        elif inp == 'e':
            break
        else:
             print (f"Gripper pos: {pos}")

        done = termination

print (f"lower_limit: {lower_limit} upper_limit: {upper_limit}")
# In the real robot we have to use a ROS interface. Disconnect the interface
# after completing the experiment.
if (not env.world.is_sim):
    env.robot_interface.disconnect()
    #env.sensor_interface.disconnect()
