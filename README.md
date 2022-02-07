# engineering_capstone

Python package that consolidates YOLOv3 object detection and OpenCV facial recognition technologies into an object retrieval program for robotic integration.

The main.py program implements object detection and facial recognition algorithms on received frames from clientdraft.py, which is run separately on a separate machine.

Steps for a quick demonstration of algorithms:

1) First, you must install an OpenCV programming environment (I used Anaconda3 for my Windows laptop to accomplish this). Guide available here: https://towardsdatascience.com/install-and-configure-opencv-4-2-0-in-windows-10-python-7a7386ae024

2) Then activate your environment with "activate <env_name>" before proceeding.

3) Then cd into OD BACKUP or FR BACKUP/src and run either "python working_OD_(BACKUP).py" or "python working_FR_(BACKUP).py"

To train the facial recognition program, upload several profiles into individual folders as shown in FR BACKUP/src/dataset1, then train the algorithm by running "python train_fr.py". Now the "working_FR_(BACKUP)" program will recognize those profiles as well.

For a fully consolidated program , refer to main.py and clientdraft.py in root directory for logic/guidance. However, the program is designed for a Raspberry Pi and Windows laptop scenario and should be altered for any other use.

Good Luck!
