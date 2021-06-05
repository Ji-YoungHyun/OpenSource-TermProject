import os

dirname= 'C:/Users/USER/Desktop/Middle_Resolution'
file_name = []
filenames = os.listdir(dirname)

for filename in filenames:
    full_filename = os.path.join(dirname, filename)
    file_name.append('C:/Users/USER/Desktop/Middle_Resolution/'+full_filename[-8:]+'/S001/L1/E01/C7.jpg')

print(file_name) # 조명 밝은 무표정 정면 사진만 뽑아냄

import math
import numpy as np
import cv2
import dlib
import imutils
from math import atan2, degrees
from math import hypot
import pandas as pd

count = pd.read_csv('C:/Users/USER/Desktop/data.csv')

def midpoint(p1, p2):
    return int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)


def eye(points):
    left_eye_lenght = distance(points[36][0], points[36][1], points[39][0], points[39][1])
    left_top = midpoint(points[37], points[38])
    left_bottom = midpoint(points[41], points[40])
    left_eye_size = hypot((left_top[0] - left_bottom[0]), (left_top[1] - left_bottom[1]))
    right_eye_lenght = distance(points[42][0], points[42][1], points[45][0], points[45][1])
    right_top = midpoint(points[43], points[44])
    right_bottom = midpoint(points[47], points[46])
    right_eye_size = hypot((right_top[0] - right_bottom[0]), (right_top[1] - right_bottom[1]))
    between_lenght = distance(points[39][0], points[39][1], points[42][0], points[42][1])
    average_lenght = (left_eye_lenght + right_eye_lenght) / 2
    average_size = (left_eye_size + right_eye_size) / 2
    if average_size > average_lenght / 3:
        print("큰 눈")
        count["0"] = count["0"] + 1
    else:
        print("작은 눈")
        count["1"] = count["1"] + 1
    if between_lenght > average_lenght * (3 / 2):
        print("넓은 미간")
        count["2"] = count["2"] + 1
    else:
        print("좁은 미간")
        count["3"] = count["3"] + 1


def eyebrow(points):
    incli = []
    angle = []
    disting = 0
    for i in LEFT_EYEBROW:
        if i == 22: continue
        incli.append(math.atan(- (points[i][1] - points[i - 1][1]) / (points[i][0] - points[i - 1][0])) / math.pi * 180)

    for i in range(len(incli)):
        if i == 0: continue
        angle.append(180 - abs(incli[i] - incli[i - 1]))

    h = (points[22][1] + points[26][1]) / 2 - points[24][1]

    if angle[1] > 160:
        disting = 1
    else:
        disting = 0

    if disting == 1:
        if incli[1] > 10:
            print('치켜 올라간 눈썹')
            count["4"] = count["4"] + 1
        elif points[26][1] > points[22][1] + 9:
            print('처진 눈썹')
            count["5"] = count["5"] + 1
        else:
            print('일자 눈썹')
            count["6"] = count["6"] + 1
        return

    if incli[1] > incli[0]:
        print('s자형 눈썹')
        count["7"] = count["7"] + 1
        return

    if angle[0] > 170:
        print('아치형 눈썹')
        count["8"] = count["8"] + 1
        return

    print('둥근 눈썹')
    count["9"] = count["9"] + 1
    return


def nose(points):
    r = (points[15][0] - points[1][0]) / 2
    face_area = math.pi * r ** 2;
    nose_area = (points[35][0] - points[31][0]) * (points[33][1] - points[28][1]) / 2

    if nose_area / face_area * 100 < 3.1:
        print('작은 코')
        count["10"] = count["10"] + 1
    elif nose_area / face_area * 100 > 3.3:
        print('큰 코')
        count["11"] = count["11"] + 1
    else:
        print('중간 코')
        count["12"] = count["12"] + 1


def angle_between(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360
    return deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)


def distance(x1, y1, x2, y2):
    result = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    return result


def mouth(points):
    # 입꼬리 쳐졌는지 올라갔는지?
    tail_angle = (angle_between(points[61], points[59], points[47]) + angle_between(points[65], points[59],
                                                                                    points[47]) +
                  angle_between(points[61], points[63], points[53]) + angle_between(points[65], points[63],
                                                                                    points[53])) / 4

    if tail_angle < 90:
        print("올라간 입꼬리")
        count["13"] = count["13"] + 1
    else:
        print("쳐진 입꼬리")
        count["14"] = count["14"] + 1

    # 윗입술, 아랫입술 크기 비슷한지 다른지?
    upper_lip_length = distance(points[50][0], points[50][1], points[61][0], points[61][1])
    lower_lip_length = distance(points[65][0], points[65][1], points[56][0], points[56][1])

    if lower_lip_length / upper_lip_length > 1.35:  # 황금비율=> 윗입술:아랫입술 = 1:1.2~1.5
        # 1.2와 1.5의 사이인 1.35로 기준 잡음
        print("윗, 아랫입술 크기 다륾")
        count["15"] = count["15"] + 1

    else:
        print("윗, 아랫입술 크기 비슷")
        count["16"] = count["16"] + 1

    # 입술 산이 뭉툭한지 뾰족한지?
    lip_mountain_angle = (angle_between(points[50], points[49], points[48]) + angle_between(points[50], points[51],
                                                                                            points[52])) / 2
    if lip_mountain_angle > 90:  # 기준: 둔각인 120도
        print("뾰족한 입술산")
        count["17"] = count["17"] + 1

    else:
        print("뭉툭한 입술산")
        count["18"] = count["18"] + 1


for i in range(len(file_name)):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('C:/Users/USER/Desktop/shape_predictor_68_face_landmarks.dat')
    image = cv2.imread(str(file_name[i]))
    image = imutils.resize(image, width=500)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    ALL = list(range(0, 68))
    JAWLINE = list(range(0, 17))
    RIGHT_EYEBROW = list(range(17, 22))
    LEFT_EYEBROW = list(range(22, 27))
    NOSE = list(range(27, 36))
    RIGHT_EYE = list(range(36, 42))
    LEFT_EYE = list(range(42, 48))
    MOUTH_OUTLINE = list(range(48, 61))
    MOUTH_INNER = list(range(61, 68))

    index = ALL


    ret, frame = cap.read()
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dets = detector(img_gray, 1)
    list_points = []

    for face in dets:
        face_img = image[face.top() - 30:face.bottom() + 30, face.left() - 30:face.right() + 30]
        face_img = cv2.resize(face_img, (600, 600))
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_dets = detector(face_gray, 1)

        for face in face_dets:
            shape = predictor(face_img, face)

            for p in shape.parts():
                list_points.append([p.x, p.y])

            list_points = np.array(list_points)

            for i, pt in enumerate(list_points[index]):
                pt_pos = (pt[0], pt[1])
                cv2.putText(face_img, str(i), pt_pos, cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0))

            print("****************얼굴 부위 별 모양입니다.****************")
            eye(list_points)
            eyebrow(list_points)
            nose(list_points)
            mouth(list_points)

    if type(list_points) is list:
        key = cv2.waitKey(1)
    else:
        key = cv2.waitKey(0)

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

print(count)
