# USAGE:
# python main.py
# client side should be run in sudo to allow access to pins?

from imutils.video import VideoStream
import matplotlib.pyplot as plt
from od_module import od_fxn
from fr_module import fr_fxn
from time import sleep
import numpy as np
import cv2
import zmq

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

# RECEIVE frames indefinitely
while True:
	frame = socket.recv_pyobj()	
	# Toggle FALSE for OD || TRUE for FR     **where Pi would inform if clawsensor ON/OFF
	
	socket.send_string('start')
	switch_response = socket.recv_string()
	if switch_response == "pressed":
		print("BUTTON PRESSED")
		clawsensor = True
	elif switch_response == "unpressed":
		clawsensor = False
	
	# claw sensor INACTIVE
	if not clawsensor: 
		# FRAME -> OD
		ODMessage = od_fxn(frame)								# Object Detection returns HIGH/LOW
		print("[INFO] sending message: {}".format(ODMessage))
		socket.send_string(ODMessage)
		cv2.imshow('OBJECT DETECTION', frame)

	# claw sensor ACTIVE
	if clawsensor:
		# FRAME -> FR
		FRMessage = fr_fxn(frame)								# Facial Recognition returns HIGH/LOW
		print("[INFO] sending message: {}".format(FRMessage))
		socket.send_string(FRMessage)
		cv2.imshow('FACIAL RECOGNITION', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'): # q closes frames
		break

cv2.destroyAllWindows()