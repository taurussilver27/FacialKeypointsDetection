# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 15:47:48 2018

@author: tomcr00se
"""

import cv2
import argparse
import csv

refPt = []
 
def click_and_record(event, x, y, flags, param):
	# grab references to the global variables
	global refPt
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt.append((x, y))
		cv2.rectangle(image, (refPt[-1][0]-2,refPt[-1][1]-2), (refPt[-1][0]+2,refPt[-1][1]+2), (0, 255, 0), 2)
		cv2.imshow("image", image)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
 
# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"], cv2.IMREAD_UNCHANGED)
# ind = image[:,:,3] == 0
# print(ind)
# image[ind] = 50
# cv2.imshow("fgj",image)
# cv2.waitKey(0)
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_record)

# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()
 
	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break

# close all open windows

# with open("glasses1.csv",'w') as myFile:
# 	wr = csv.writer(myFile)

print(refPt)
myFile = open('face2.csv.csv', 'w')
wr = csv.writer(myFile)
wr.writerows([refPt])
myFile.flush()
myFile.close()
cv2.destroyAllWindows()

 