import pickle
from imutils import face_utils
import imutils
import time
import dlib
import cv2
from imutils.video import VideoStream
import face_recognition
from datetime import datetime

now = datetime.now()
today = now.strftime("%m/%d/%Y-%H:%M:%S")

data = pickle.loads(open("encodings.pickle", "rb").read())
# load dlib's HOG + Linear SVM face detector
print("[INFO] loading HOG + Linear SVM face detector...")
detector = dlib.get_frontal_face_detector()
# load the input image from disk, resize it, and convert it from
# BGR to RGB channel ordering (which is what dlib expects)
vs = VideoStream(src=0).start()
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# perform face detection using dlib's face detector
    start = time.time()
    print("[INFO[ performing face detection with dlib...")
    rects = detector(rgb, 2)
    end = time.time()
    print("[INFO] face detection took {:.4f} seconds".format(end - start))

# convert the resulting dlib rectangle objects to bounding boxes,
# then ensure the bounding boxes are all within the bounds of the
# input image
    boxes = []
    #boxes = [(y, x + w, y + h, x) for (x, y, w, h) in  enumerate(rects)]
    for (i, rect) in enumerate(rects):
 	 # convert dlib's rectangle to a OpenCV-style bounding box
 	 # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        
        boxe = (y, x + w, y + h, x)
        boxes.append(boxe)
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = [] 
    for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding,0.4)
        name = "Unknown"

		# check to see if we have found a match
        if True in matches:
			# find the indexes of all matched faces then initialize 
			# dictionary to count the total number of times each face
			# was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
            name = max(counts, key=counts.get)
		
		# update the list of names
        names.append(name)

	# loop over the recognized face
    for ((top, right, bottom, left), name) in zip(boxes, names):
		# draw the predicted face name on the image
        if name == "Unknown":
            now = datetime.now()
            frame_crop = frame[top:bottom, left:right]
            today = now.strftime("%m_%d_%Y   %H.%M.%S")
            cv2.imshow("Frameframe_crop", frame_crop)
            cv2.imwrite('data/'+today+'.jpg',frame_crop)
        cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# show the output image
cv2.destroyAllWindows()
vs.stop()  