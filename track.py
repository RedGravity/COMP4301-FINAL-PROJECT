# import the necessary packages
import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils

# Set HOG Descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

vid = cv2.VideoCapture('sun.mp4')
maxWidth = 400
maxHeight = 300

# cv.VideoWriter(filename, apiPreference, fourcc, fps, frameSize[, isColor]	)
vid_out = cv2.VideoWriter('trackedVideo.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, (maxWidth, maxHeight))

while(True):
	# Loops runs on each frame of input video
	r, frame = vid.read()
	if r == True:
		# Resize the frame to a max of 350 pixels in frame_width,
		# Then convert to greyscale to speed up calculations
		frame = cv2.resize(frame, (maxWidth, maxHeight))
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		'''
		# win-stride is step size in the x,y directions of our window
		rects, weights = hog.detectMultiScale(gray_frame, winStride=(7,7) )


		# draw boxes on frame
		for (x, y, w, h) in rects:
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		'''



		# detect people in the image
		(rects, weights) = hog.detectMultiScale(gray_frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

		# apply non-maxima suppression to the bounding boxes using a
		# fairly large overlap threshold to try to maintain overlapping
		# boxes that are still people
		rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
		pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

		# draw the final bounding boxes
		for (xA, yA, xB, yB) in pick:
			#cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
			cv2.rectangle(frame, (xA, yA), (xB, yB), (255, 0, 0), 2)




		# add frame to output video file
		vid_out.write(frame)
	else:
		break

vid.release()
vid_out.release()
