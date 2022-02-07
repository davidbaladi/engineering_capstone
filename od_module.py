# Required files in same directory: 'COCO', 'weights', 'cfg'

import cv2
import numpy as np


classes = []
font = cv2.FONT_HERSHEY_SIMPLEX

# load YOLO/tiny
net = cv2.dnn.readNet("weights/yolov3-tiny.weights", "cfg/yolov3-tiny.cfg")		# Load YOLO / YOLO-tiny
						# tiny = less detection - but FASTER

def od_fxn(frame): #datatyp of frame?
	message = "OD_running..."
	with open("coco/coco.names", "r") as f: # Reading names of classes						# DIRECTORY IMPORTANT
		classes = [line.strip() for line in f.readlines()]
	layer_names = net.getLayerNames()
	output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))

#	frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
	height, width, channels = frame.shape

	# DETECTING
	blob = cv2.dnn.blobFromImage(frame,0.00392,(320,320),(0, 0, 0), True, crop=False)
											   # ^   ^  smaller = faster


#   CHANNELS:
#	for b in blob:
#		for n, img_blob in enumerate(b):
#			cv2.imshow(str(n), img_blob)

		
	confidences = []
	class_ids = []
	boxes = []
	# Each blob -> algorithm	
	net.setInput(blob)
	outs = net.forward(output_layers)
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]

			if confidence > 0.4:
				# Object Detected!!
				center_x = int(detection[0]*width)
				center_y = int(detection[1]*height)
				w = int(detection[2]*width)
				h = int(detection[3]*height)
				x = int(center_x - w/2)
				y = int(center_y - h/2)
					
				boxes.append([x,y,w,h])
				confidences.append(float(confidence))
				class_ids.append(class_id)
#				number_of_objects_detected = len(boxes)
#				print(number_of_objects_detected)
					
				indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#				print(indexes) # For checking for duplicate objects (~30 mins)
					
					
				for i in range(len(boxes)):
					if i in indexes:
						x,y,w,h = boxes[i]
						label = str(classes[class_ids[i]])
#						print(label)
						confidence = confidences[i]
						color = colors[class_ids[i]]


#####					# TARGET COLOR
						if label == "person":
							color = (0,0,255)
							# Send text message over socket
							message = "object"
							
						cv2.rectangle(frame, (x,y), (x + w, y + h), color, 2)
						cv2.putText(frame, label + str(round(confidence, 2)) + "%", (x, y + 33), font, 3, color, 3)

	return message