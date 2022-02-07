# FOLDER NAME = label

import cv2
import os
import numpy as np
from PIL import Image
import pickle


# Get directory PATH of *this* file location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Look for FOLDER in BASE_DIR
image_dir = os.path.join(BASE_DIR, "dataset")

# import cv2 for training roi as in fr.py
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids = {}

y_labels = []
x_train = []


# Finds all PNGs / JPGs in FOLDER
for root, dirs, files in os.walk(image_dir): # Search each element in FOLDER
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)

			# Grabbing FOLDER NAME (aka label)
			label = os.path.basename(os.path.dirname(path)).replace("", "-").lower()
										# root ^^  could be a solution to an error	 
			#print(label, path)

			# Creating ids for labels  -> Label IDs
			if not label in label_ids:
				label_ids[label] = current_id
				current_id +=1
			id_ = label_ids[label]
			#print(label_ids)


			#y_labels.append(label) #converts label to number
			#x_train.append(path) #verify image and turn into numpy array, GRAY
			
			pil_image = Image.open(path).convert("L") # Frames to grayscale (L)
			
			# EXTRA ADJUSTMENTS to make recognition better (~54 mins)
			size = (550, 550)
			final_image = pil_image.resize(size, Image.ANTIALIAS)


			image_array = np.array(pil_image, "uint8") # Grayscale to numpy array
			#print(image_array) # Prints numpy arrays; converted frames to vectors for training
			

			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5) # SclFac too high not good
			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id_)

#print(y_labels)
#print(x_train)

# Saving LABEL IDs using PICKLE
with open("labels.pickle", 'wb') as f:  # wRITING bITES to fILE
	pickle.dump(label_ids, f) # IDs to fILE

		

		# !!!! TRAINING OpenCV RECOGNIZER !!!!
				# lbph
print("Training ...")
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")
print("... Complete")