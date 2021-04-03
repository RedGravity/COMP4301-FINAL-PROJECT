# import the necessary packages
import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils

print("type 'v' to generate a video file with people detection, or type 'cam' to use people detection on your webcam")
type = input()


if(type == "v"):
	print()
	print("input the name of the video file you want to run the program on (make sure your video file is in your working directory):")

	file_name = input()

	print()
	print("generating new video file with people detection")

	# Create a HOG Descriptor
	hog = cv2.HOGDescriptor()

	# Make the Supprot Vector Machine able to classify people
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	# Create an object to store information from the video file specified at program start
	vid = cv2.VideoCapture(file_name)

	# Set width and height in pixels of output video file
	maxWidth = 400
	maxHeight = 300

	# cv.VideoWriter(filename, apiPreference, fourcc, fps, frameSize[, isColor]	)
	vid_out = cv2.VideoWriter('trackedVideo.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, (maxWidth, maxHeight))

	# Runs on each frame of input video file
	while(True):
		r, frame = vid.read()
		if r == True:
			# Resize the frame to maxWidth by maxHeight in pixels,
			frame = cv2.resize(frame, (maxWidth, maxHeight))

			# Then convert to greyscale to speed up calculations
			gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# Detect people in the image
			(rects, weights) = hog.detectMultiScale(gray_frame, winStride=(4, 4), padding=(8, 8), scale=0.8)

			# Use Non Maximum Suppression to reduce number of bounding boxes, to just the best fit
			rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
			pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

			# draw the final bounding boxes
			for (xa, ya, xb, yb) in pick:
				#cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
				cv2.rectangle(frame, (xa, ya), (xb, yb), (255, 0, 0), 2)


			# add frame to output video file
			vid_out.write(frame)
		else:
			break

	vid.release()
	vid_out.release()
	print()
	print("your video with people detection can be found in your working directory and is named trackedVideo.avi")



elif(type == 'cam'):
	# Set HOG Descriptor
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	vid = cv2.VideoCapture(1)
	maxWidth = 250
	maxHeight = 380


	while(True):
		# Loops runs on each frame of input video
		r, frame = vid.read()
		if r == True:
			# Resize the frame to a max of 350 pixels in frame_width,
			# Then convert to greyscale to speed up calculations
			frame = cv2.resize(frame, (maxHeight, maxWidth))
			gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


			# detect people in the image
			(rects, weights) = hog.detectMultiScale(gray_frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

			# apply non-maxima suppression to the bounding boxes using a
			# fairly large overlap threshold to try to maintain overlapping
			# boxes that are still people
			rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
			pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

			# draw the final bounding boxes
			for (xa, ya, xb, yb) in pick:
				#cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
				cv2.rectangle(frame, (xa, ya), (xb, yb), (255, 0, 0), 2)

			cv2.imshow('output', frame)
			c = cv2.waitKey(1)
			if c == 27:
				break
		else:
			break

	vid.release()



else:
	print("invalid entry. run the program again and enter a correct string when prompted")
