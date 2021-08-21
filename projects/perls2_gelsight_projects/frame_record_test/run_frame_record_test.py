"""Project template example running the environment.
"""
from __future__ import division
import time
from frame_record_test import FrameRecordTest
import logging
import cv2
logging.basicConfig(level=logging.DEBUG)


BASE_FILENAME = "test_recordings/episode_"

def get_action(observation):
    """Run your policy to produce an action.
    """
    action = [0, 0, 0, 0, 0, 0]
    return action


env = FrameRecordTest('frame_record_test.yaml', True, "TemplateEnv")

for ep_num in range(10):
    logging.debug('episode ' + str(ep_num - 1) + ' complete...pausing...')
    step = 0
    observation = env.reset()
    done = False

    filename = BASE_FILENAME + str(ep_num) + ".avi"
    out_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (400, 300))

    while not done:
        start = time.time()

        frame = observation["left_frame"]
        out_writer.write(frame)

        action = get_action(observation)

        # Pass the start time to enforce policy frequency.
        observation, reward, termination, info = env.step(action, start=start)

        done = termination
    
    out_writer.release()

# In the real robot we have to use a ROS interface. Disconnect the interface
# after completing the experiment.
if (not env.world.is_sim):
    env.robot_interface.disconnect()
    env.sensor_interface.disconnect()
