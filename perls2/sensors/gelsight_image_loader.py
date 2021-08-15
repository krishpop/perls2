import cv2
import numpy as np
from threading import Thread
import csv
import redis
import struct
from gelsight.gelsight_driver import GelSight
from gelsight.util.processing import warp_perspective

REDIS_HOST = "172.16.0.4" #TODO: Get this info from available redis configs
REDIS_PORT = "6379"

LEFT_IMG_VAR = "left_gel_img"
RIGHT_IMG_VAR = "right_gel_img"

LEFT_GEL_ADDR = "http://169.254.165.13"
RIGHT_GEL_ADDR = None

def read_csv(filename="gelsight_config.csv"):
	rows = []

	with open(filename, "r") as csvfile:
		csvreader = csv.reader(csvfile)
		header = next(csvreader)
		for row in csvreader:
			rows.append((int(row[1]), int(row[2])))

	return rows


corners = tuple(read_csv())
left_gs = GelSight(
	IP=LEFT_GEL_ADDR,
	corners=corners,
	tracking_setting=None,
	output_sz=(400, 300),
	id="right",
)
left_gs.start()

"""right_gs = GelSight(
	IP=LEFT_GEL_ADDR,
	corners=corners,
	tracking_setting=None,
	output_sz=(400, 300),
	id="right",
)
right_gs.start()"""

def encoded_image(img, redisClient, time_sec, time_micro):
	"""Returns the image and metadata encoded in a byte string"""
	shape = img.shape
	meta_data_str = struct.pack(">IIII", time_sec, time_micro, shape[0], shape[1])

	image_str = img.tobytes()
	total_str = meta_data_str + image_str

	return total_str


def main():
	redisClient = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
	while True:
		left_img = left_gs.stream.image
		left_sec, left_micro = redisClient.time()

#		right_img = right_gs.stream.image
#		right_sec, right_micro = redisClient.time()

		left_img = warp_perspective(left_img, left_gs.corners, left_gs.output_sz)
#		right_img = warp_perspective(right_img, right_gs.corners, right_gs.output_sz)
		
		left_enc = encoded_image(left_img, redisClient, left_sec, left_micro)
#		right_enc = encoded_image(right_img, redisClient, right_sec, right_micro)

		redisClient.set(LEFT_IMG_VAR, left_enc)
#		redisClient.set(RIGHT_IMG_VAR, right_enc)


if __name__ == "__main__":
	main()

