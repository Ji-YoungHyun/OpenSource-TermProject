import math
import numpy as np
import cv2
import dlib
import imutils
from math import atan2, degrees
from math import hypot
from PIL import ImageFont, ImageDraw, Image
from wordcloud import WordCloud
from konlpy.tag import Okt
from collections import Counter
import matplotlib.pyplot as plt

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
image = cv2.imread('image.jpg')
image = imutils.resize(image, width=500)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
result_text = []

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
        result_text.append('이성들에게 인기가 많지만 유혹에 약하다. 연인관계를 조심할 필요가 있다.')
    else:
        print("작은 눈")
        result_text.append('실속이 많고 빈틈이 많이 없지만 내성적이고 고집이 세다.')

    if between_lenght > average_lenght * (3 / 2):
        print("넓은 미간")
    else:
        print("좁은 미간")

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
            result_text.append('개성이 강하고 경쟁을 즐긴다. 승부욕이 강해 라이벌이 있으면 더 열심히 한다. 집중력과 추진력도 뛰어나다.')

        elif points[26][1] > points[22][1] + 9:
            print('처진 눈썹')
            result_text.append('부드럽고 젠틀한 성격으로 친구를 빨리 만든다. 감정 이입을 잘해 다른 사람의 문제 해결에도 잘 나선다.')

        else:
            print('일자 눈썹')
            result_text.append('당당하고 지혜로우며 신속하게 문제를 해결한다. 고집은 조금 세지만 목표를 이루므로 신뢰도가 높다.')

        return

    if incli[1] > incli[0]:
        print('s자형 눈썹')
        result_text.append('자신감이 넘치고 호쾌한 성격으로 성공할 가능성이 크다. 사리분별이 명확하고 티 나는 거짓말을 싫어한다.')
        return

    if angle[0] > 170:
        print('아치형 눈썹')
        result_text.append('매우 사교적이고 주변 사람들을 끌어들이는 매력의 소유자. 노력하지 않아도 존재감이 있다.')
        return

    print('둥근 눈썹')
    result_text.append('공정하고 논리적이다. 주변 사람에게 책임을 묻지 않고, 감정에 쉽게 휩쓸리지 않는다.')
    return


def nose(points):
    r = (points[15][0] - points[1][0]) / 2
    face_area = math.pi * r ** 2;
    nose_area = (points[35][0] - points[31][0]) * (points[33][1] - points[28][1]) / 2

    if nose_area / face_area * 100 < 3.1:
        print('작은 코')
        result_text.append('자신감이 부족하다. 매사에 주저를 많이 하고 모처럼 성사될 일도 주저하여 실패로 끝나는 경우가 있다.')

    elif nose_area / face_area * 100 > 3.3:
        print('큰 코')
        result_text.append('자기 중심적이다. 자아가 지나치게 강해 대인관계에서 상대를 배려하지 못하고 고독해지는 경우가 있다.')

    else:
        print('중간 코')
        result_text.append('')


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
        result_text.append('긍정적이면서 쾌활한 성격. 평소에도 미소를 띤 인상으로 대부분 대인관계가 좋은 편이다. 성실한 업무 자세로 윗사람에게 좋은 평을 듣는다.')
    else:
        print("쳐진 입꼬리")
        result_text.append('방어적이고 겁이 많은 편. 세상에 상처받는 것을 두려워하며 적은 사람과 깊은 관계를 유지한다.')

    # 윗입술, 아랫입술 크기 비슷한지 다른지?
    upper_lip_length = distance(points[50][0], points[50][1], points[61][0], points[61][1])
    lower_lip_length = distance(points[65][0], points[65][1], points[56][0], points[56][1])

    if lower_lip_length / upper_lip_length > 1.35:  # 황금비율=> 윗입술:아랫입술 = 1:1.2~1.5
        # 1.2와 1.5의 사이인 1.35로 기준 잡음
        print("윗, 아랫입술 크기 다륾")
        result_text.append('모든 사물에 대해 노력과 애정을 쏟아 붓는 타입. 그러나 유혹에 매우 약하니 조심할 것.')
    else:
        print("윗, 아랫입술 크기 비슷")
        result_text.append('엄격하면서도 섬세한 성격의 타입으로 자존심과 자기 주장이 강하다. 부도덕한것을 매우 싫어한다.')

    # 입술 산이 뭉툭한지 뾰족한지?
    lip_mountain_angle = (angle_between(points[50], points[49], points[48]) + angle_between(points[50], points[51],
                                                                                            points[52])) / 2
    if lip_mountain_angle > 120:  # 기준: 둔각인 120도
        print("뾰족한 입술산")
        result_text.append('신경이 예민하고 솔직해 직설적인 말을 잘 뱉음. 반면 내면은 감성적이고 여려 눈물이 많다.')
        result_text.append('일을 척척 열심히 하는 노력형. 어떠한 것에 대해 자신의 의견을 확실히 하지 않으면 성이 안차는 고집이 있다.')

    else:
        print("뭉툭한 입술산")
        result_text.append('남의 시선을 의식하지 않고 하고픈 대로 움직인다. 털털하고 성격이 좋아 인기가 많은 편이다.')



while True:
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
            cv2.imshow('face', face_img)

            print("****************얼굴 부위 별 모양입니다.****************")
            eye(list_points)
            eyebrow(list_points)
            nose(list_points)
            mouth(list_points)

    # cv2.imshow('result', image)

    result_text = "\n".join(result_text)
    print()
    print("****************관상 결과 입니다.****************")
    print(result_text)
    img = np.zeros((500, 1000, 3), np.uint8)
    b, g, r, a = 255, 255, 255, 0
    fontpath = "NanumSquare_acB.ttf"
    font = ImageFont.truetype(fontpath, 20)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((60, 70), result_text, font=font, fill=(b, g, r, a))
    img = np.array(img_pil)
    cv2.imshow("res", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    sentences_tag = []
    okt = Okt()
    sentences_tag = okt.pos(result_text)
    noun_adj_list = []
    for word, tag in sentences_tag:
        if tag in ['Noun', 'Adjective']:
            noun_adj_list.append(word)
    counts = Counter(noun_adj_list)
    tags = counts.most_common(10)
    
    wordcloud = WordCloud(font_path="C:/Users/USER/Desktop/NanumSquare_acB.ttf", background_color= "white", width=1000, height=1000, max_words=10, max_font_size=300).generate(str(tags))
    plt.figure(figsize=(10, 8))
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.show()

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
