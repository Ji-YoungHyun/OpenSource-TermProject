import numpy as np
import cv2
import dlib
import imutils
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
image = cv2.imread('C:/Users/USER/Desktop/1.jpg')
image = imutils.resize(image, width=500)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))

index = ALL

def split_face(image, detector, predictor):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(img_gray, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(img_gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            clone = image.copy()
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2)

            for (x, y) in shape[i:j]:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

            (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))

            roi = image[y:y + h, x:x + w]
            roi = imutils.resize(roi, width=250, inter = cv2.INTER_CUBIC)

            cv2.imshow("ROI", roi)
            cv2.imshow("Image", clone)
            cv2.waitKey(0)

    output = face_utils.visualize_facial_landmarks(image, shape)
    cv2.imshow("Image", output)
    cv2.waitKey(0)

split_face(image, detector, predictor)

while True:
    ret, frame = cap.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = detector(img_gray, 1)

    for face in dets:
        face_img = frame[face.top() - 30:face.bottom() + 30, face.left() - 30:face.right() + 30]
        face_img = cv2.resize(face_img, (600, 600))
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_dets = detector(face_gray, 1)

        for face in face_dets:
            shape = predictor(face_img, face)

            list_points = []

            for p in shape.parts():
                list_points.append([p.x, p.y])

            list_points = np.array(list_points)

            for i, pt in enumerate(list_points[index]):
                pt_pos = (pt[0], pt[1])
                cv2.putText(face_img, str(i), pt_pos, cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0))
            cv2.imshow('face', face_img)
        # cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 3)
    # cv2.putText(frame, 'hello', (300, 100), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0))
    cv2.imshow('result', frame)

    # cv2.imshow('face', face_img)
    key = cv2.waitKey(1)

    if key == 27:
        break
    elif key == ord('1'):
        index = ALL
    elif key == ord('2'):
        index = LEFT_EYEBROW + RIGHT_EYEBROW
    elif key == ord('3'):
        index = LEFT_EYE + RIGHT_EYE
    elif key == ord('4'):
        index = NOSE
    elif key == ord('5'):
        index = MOUTH_OUTLINE + MOUTH_INNER
    elif key == ord('6'):
        index = JAWLINE

cv2.destroyAllWindows()
cap.release()
