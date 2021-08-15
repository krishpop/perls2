from perls2.sensors.camera_interface import CameraInterface
import numpy as np
import logging
from abc import ABCMeta, abstractmethod
import cv2
import redis
import time
import struct
import csv
from perls2.redis_interfaces.redis_keys import *

class GelSightCameraInterface(CameraInterface):
	"""GelSight Camera Class that makes the left and 
		right images of the GelSight Accessible"""

	REDIS_HOST = "172.16.0.4" #TODO: Get this info from available redis configs
	REDIS_PORT = "6379"

	LEFT_IMG_VAR = "left_gel_img"
	RIGHT_IMG_VAR = "right_gel_img"

	LEFT_GEL_ADDR = "http://169.254.165.13"
	RIGHT_GEL_ADDR = None

	def __init__(self, gel_cfg="gelsight_config.csv", config=None):
		self.config = config
		if self.config:
			pass

		self.gel_cfg = gel_cfg
		self.redisClient = redis.Redis(host=self.REDIS_HOST, port=self.REDIS_PORT)


	def start(self):
		pass

	def stop(self):
		pass

	def reset(self):
		self.stop()
		self.start()

	def decode_image_str(self, enc_img_str):
		"""Given an image and metadata encoded as a byte string, return the image as a numpy array"""
		time_sec, time_micro, h, w = struct.unpack(">IIII", enc_img_str[:16])
		unp_image = np.frombuffer(enc_img_str, dtype=np.uint8, offset=16).reshape(h, w, 3)

		return time_sec, time_micro, unp_image
	
	def _get_frame(self, img_var):
		ret_str = self.redisClient.get(img_var)
		time_sec, time_micro, img = self.decode_image_str(ret_str)
		return time_sec, time_micro, img

	def frames(self):
		left_time_sec, left_time_micro, left_img = self._get_frame(self.LEFT_IMG_VAR)
		#right_time_sec, right_time_micro, right_img = self._get_frame(RIGHT_IMG_VAR)

		return (left_time_sec, left_time_micro, left_img)#, (right_time_sec, right_time_micro, right_img)
		
if __name__ == "__main__":
	gs_interface = GelSightCameraInterface()
	while True:
		frames = gs_interface.frames()

		img = frames[2]   	

		cv2.imshow("Left Camera", img)
		if cv2.waitKey(1) == 27:
			break
	
	cv2.destroyAllWindows()


