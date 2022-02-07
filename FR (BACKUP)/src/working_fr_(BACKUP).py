# sudo tag for GPIO access
# works on Pi 

import numpy as np 
import cv2
import pickle
#from gpiozero import LED
from time import sleep

#led = LED(23)

# This cascade requires (works with) gray frames
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {}
with open("labels.pickle", 'rb') as f:  # rEAD bITES from fILE
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()} # 180 dictionary switch keys and values


# FOR Pi Camera:
#cap = VideoStream(usePiCamera=True).start()

# FOR Windows Webcam:
cap = cv2.VideoCapture(0)

while True:

	ret, frame = cap.read() # Reads each frame of video capture
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# DOES THE MAGIC
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5) # SclFac too high not good
	
	for (x, y, w, h) in faces:
#		print(x,y,w,h) # Values from faces instances

		################ for correct face only, send signal

		roi_gray = gray[y:y+h, x:x+w]

#		# !!!! RECOGNIZER !!!! (deep learned model: keras/tensorflow/pytorch/scikit_learn etc.)
		id_, conf = recognizer.predict(roi_gray) # Extracting values; Prediction on ROI only
#	FYI: confidence lvl varies A LOT - even weird numbers with perfect facial images
		if conf >= 35 and conf <= 95:
			print(id_) # But not LABEL yet :(    so get from pickle!
			print(labels[id_])
			
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_] # Gets LABEL
			color = (255, 255, 255)
			stroke = 2
			cv2.putText(frame, name, (x,y + 30), font, 1, color, stroke, cv2.LINE_AA)
#			led.on()
#		else:
#			led.off()

                                               ############
#


		# Saving face part of frame
		img_item = "face_captured.png"
		cv2.imwrite(img_item, roi_gray)

		# Drawing rectangles
		color = (0, 255, 0) #BGR 0-255
		stroke = 3 # Line thickness
		end_cord_x = x + w
		end_cord_y = y + h
		cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke) # Drawing on original frame

	# Display frames
	cv2.imshow('FACIAL RECOGNITION', frame) # DO NOT imgshow
#	cv2.imshow('gray', gray) # copy&paste above line, change BOTH params to generate additional frames/windows
	if cv2.waitKey(5) & 0xFF == ord('q'): # q closes frames
		break

# Releases capture when everything done
cap.release()
cv2.destroyAllWindows()