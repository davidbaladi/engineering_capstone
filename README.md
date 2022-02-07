# engineering_capstone

Python package that consolidates YOLOv3 object detection and OpenCV facial recognition technologies into an object retrieval program for robotic integration.

The main.py program consolidates working object detection and facial recognition.

Steps for a quick demonstration:

1) First, you must install an OpenCV programming environment using Anaconda3 on your machine. Guide available here: https://towardsdatascience.com/install-and-configure-opencv-4-2-0-in-windows-10-python-7a7386ae024

2) Then activate your environment with "activate <env_name>"

3) Then cd into OD BACKUP or FR BACKUP/src and run either "python working_OD(BACKUP).py" or "python working_FR(BACKUP).py"

To train the facial recognition program, upload profiles into individual folders as shown in FR BACKUP/src/dataset1, then train the algorithm by running "python train_fr.py". Now the "working_FR(BACKUP)" program will recognize those profiles as well.

For a fully consolidated program , refer to main.py and clientdraft.py in root directory.
