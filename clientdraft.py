# USAGE:
# sudo python custom_client.py --server-ip SERVER_IP --server-port 5555
								# (or try 'localhost' or loopback IP)
								# (check ipconfig terminal command on server machine)
from imutils.video import VideoStream
import RPi.GPIO as GPIO
from time import sleep
import zmq
import cv2

GPIO.setmode(GPIO.BCM)
GPIO.setup(17,  GPIO.OUT)
GPIO.setup(23,  GPIO.OUT)
GPIO.setup(24,  GPIO.OUT)
GPIO.setup(16,  GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://192.168.2.42:5555") # 192.168.2.42 is server ip address using homo wifi
vs = VideoStream(usePiCamera=True, resolution=(640, 480)).start() # Specifying resolution saves CPU cycles
sleep(1)

counter = 0
ocounter = 0
fcounter = 0
# SEND frames indefinitely
while True:

	frame = vs.read()
	socket.send_pyobj(frame)
	
	empty = socket.recv_string()

	if (GPIO.input(16)):
		message = "pressed"
		print("button pressed")
	else:
		message = "unpressed"
	
	socket.send_string(message)
	response = socket.recv_string()
	print("[INFO] received reply message: '{}'".format(response))

	if response == "object":
		print("Found Target Object! Approaching Object ... ")
		ocounter += 1
		if ocounter > 10:
			GPIO.cleanup()
			sleep(1)
			GPIO.output(24, GPIO.HIGH)
			counter = 0
			fcounter = 0
			GPIO.cleanup()
			

	elif response == "face":
		print("Found Target Person! Returning Object ...")
		fcounter += 1
		if fcounter > 10:
			GPIO.cleanup()
			sleep(1)
			GPIO.output(17, GPIO.HIGH)
			counter = 0
			ocounter = 0
			GPIO.cleanup()

	else:
		counter += 1
		if counter > 15:
			GPIO.cleanup()
			sleep(1)
			GPIO.output(23, GPIO.HIGH)
			ocounter = 0
			fcounter = 0
			print("Looking for object...")
			GPIO.cleanup()
			