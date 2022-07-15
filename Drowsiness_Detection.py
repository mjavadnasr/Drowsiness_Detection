#import libraries
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
#eye_ratio function for calculate EAR
def eye_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear	
#declare default variables
max_ear = 0.25
max_frame = 20
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

#define left eye and right eye range in model
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
#define first video capture that cv2 find
capture=cv2.VideoCapture(0)
#frame counter
counter=0
while True:
	ret, frame=capture.read()
    #resize frame
	frame = imutils.resize(frame, width=450)
    #grayScalize frame
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = detector(gray, 0)
	for face in faces:
        #set shape with aspects of face in grayScale
		shape = predictor(gray, face)
        #use numPy for shape
		shape = face_utils.shape_to_np(shape)
        #use dlib for finding eye aspect
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
        #calculate EAR
		leftEAR = eye_ratio(leftEye)
		rightEAR = eye_ratio(rightEye)
        #mid of Ear
		ear = (leftEAR + rightEAR) / 2.0
        #draw convex point arround eye
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        #main program
		if ear < max_ear:
			counter += 1
			print (counter)
			if counter >= max_frame:
				cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				print ("Drowsy")
		else:
			counter = 0
	cv2.imshow("Frame", frame) 
	if cv2.waitKey(1) == ord("q"):
		break
cv2.destroyAllWindows()
capture.release() 
