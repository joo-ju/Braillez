import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
from braille import table_kor
from unicode import join_jamos

# 변수명, 주석, print 수정 완료

# 카메라로 사진을 다시 찍어서 실행해봄. 명암이 균일하지 않아서 검출에 힘들었던 사진말고 조금더 명암이 균일한 조건에서 촬영함
# 조건 : 표지판과 카메라 10cm 거리, 카메라와 표지판이 평행하게 촬영

# imgBGR = cv.imread('./data/smoking.png')
# imgBGR = cv.imread('./data/smoking_2.jpg')
# imgBGR = cv.imread('./data/smoking_3.png')
# imgBGR = cv.imread('./data/g.jpg')
# imgBGR = cv.imread('./data/f.jpg')
# imgBGR = cv.imread('./data/e.jpg')
# imgBGR = cv.imread('./data/k.jpg')
# imgBGR = cv.imread('./data/m.jpg')
# imgBGR = cv.imread('./data/o.jpg')
# imgBGR = cv.imread('./data/n.jpg')
imgBGR = cv.imread('./data/ab.jpg')
# imgBGR = cv.imread('./data/j.jpg')


# by 김주희_이미지 사이즈 변경_201020
imgBGR = cv.resize(imgBGR, (2592, 1944))

imgYUV = cv.cvtColor(imgBGR, cv.COLOR_BGR2YUV)

# 반전(검->흰, 흰->검)
# y의 형식을 imgYUV와 같게 해주기 위해 실행
y = imgYUV

# 밝기를 나타내는 y layer으로만 구성
y[:, :, 0] = imgYUV[:, :, 0]
y[:, :, 1] = imgYUV[:, :, 0]
y[:, :, 2] = imgYUV[:, :, 0]


#201007_이진 영상에서 점자 검출
ret,thr = cv.threshold(y,210,255,cv.THRESH_BINARY)

kernel = np.ones((3, 3), np.uint8)
imgClosing = cv.morphologyEx(thr, cv.MORPH_CLOSE, kernel)

# by 배아랑이_201012
# 이미지의 가장자리 일정 부분은 하얗게 만든다. -> 그림 잘라 버리기
# 그림의 가로세로 길이의 8분의 1 정도만 가장자리를 검게 바꿔준다.
cutEdge = [(int)(imgClosing.shape[0]/8), (int)(imgClosing.shape[1]/8)]
imgClosing[:cutEdge[0], :] = 125
imgClosing[-cutEdge[0]:, :] = 125
imgClosing[:, :cutEdge[1]] = 125
imgClosing[:, -cutEdge[1]:] = 125

# by 김주희_contouring하여 가로 세로 비율과 크기를 이용하여 점들을 검출한다._200702
contours, _ = cv.findContours(imgClosing[:, :, 0], cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
center=[]
for i in range(len(contours)):
    cnt = contours[i]
    area = cv.contourArea(cnt)
    x, y, w, h = cv.boundingRect(cnt)

    # by 김주희_contour 면적 _200702
    rect_area = w * h

    # by 김주희_가로 세로의 비율 _200702
    aspect_ratio = float(w)/h

    # by 김주희_컨투어한 영역의 비율을 보고 사각형을 그림 _200702
    #   점자에 대한 contour를 찾는 과정_가로 세로 비율이 1:1에서 크게 벗어난 것을 제외하고 표시
    if (aspect_ratio > 0.8) and (aspect_ratio < 1.2):
        # 크기가 10픽셀 이하인 것은 모두 없애기._201012
        if (w > 5) and (w < 25):
            cv.rectangle(imgClosing, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cx = int(x + w/2)
            cy = int(y + h/2)
            mnt = cv.moments(cnt)

            # by 김주희_컨투어한 영역의 중심점이 아닌 사각형의 중심점을 표시함._201012
            cv.circle(imgClosing, (cx, cy), 3, (255, 0, 0), -1)
            center.append([cx, cy])


# by 김주희_점자 중 가장 처음의 2점 혹은 가장 마지막의 2점에 대한 기울기 _201012
dx2 = center[0][0] - center[5][0]
dy = center[0][1] - center[5][1]


# by 김주희_두 기울기의 평균으로 회전하기_201019
ave = math.atan(dy/dx2)

# by 김주희_ 각도 구하기 _201012
angle = ave * (180.0 / math.pi)

# by 김주희_ 각도만큼 회전시켜 imgRotate 영상 만들기 _201012
h, w = imgClosing.shape[:2]
M = cv.getRotationMatrix2D((w/2, h/2), angle, 1)
imgRotate = cv.warpAffine(imgClosing, M,(w, h))

# by 김주희_ 현재 imgClosing에 contour 실시함 -> 사각형 그려져 있음._201012
# by 김주희_ 거리를 계산하여 주변에 점이 많으면 같이 없앰.
rotateCenter = []
centerX = []
centerY = []

# by 김주희_ 회전된 영상에서 점자를 구성하는 점을 검출  _201012
contours, _ = cv.findContours(imgRotate[:, :, 0], cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
for i in range(len(contours)):
    cnt = contours[i]
    area = cv.contourArea(cnt)
    x, y, w, h = cv.boundingRect(cnt)

    # by 김주희_contour 면적 _200702
    rect_area = w * h

    # by 김주희_가로 세로의 비율 _200702
    aspect_ratio = float(w)/h


    # by 김주희_컨투어한 영역의 비율을 보고 사각형을 그림 _200702
    #   점자에 대한 contour를 찾는 과정_가로 세로 비율이 1:1에서 크게 벗어난 것을 제외하고 표시
    # if(aspect_ratio>0.8)and(aspect_ratio<1.5)and(f <= float(f)*0.7):

    # cv.rectangle(imgClosing, (x, y), (x + w, y + h), (0, 127, 127), 10)
    if (aspect_ratio > 0.8) and (aspect_ratio < 1.2):
        # 크기가 10픽셀 이하인 것은 모두 없애기._201012
        if (w > 5) and (w < 25):
            cv.rectangle(imgRotate, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cx = int(x + w/2)
            cy = int(y + h/2)
            mnt = cv.moments(cnt)

            # by 김주희_컨투어한 영역의 중심점이 아닌 사각형의 중심점을 표시함._201012
            cv.circle(imgRotate, (cx, cy), 3, (255, 0, 0), -1)
            rotateCenter.append([cx, cy])
            centerX.append(cx)
            centerY.append(cy)




#by 김주희_ _201019
centerX.sort()
centerY.sort()
hline = []
vline = []

sameLevelY = []
sameLevelX = []
temp = []
temp2 = []
midX = []
midY = []
index = 0   # sameLevel 변수의 레벨을 나누는 것. (같은 라인인지 아닌지 구분)

# by 김주희_ 거리를 구할 때 인덱스오류가 나서 하나 더 대입해줌 - 영향 없음_201019
centerY.append(0)
centerX.append(0)

# by 김주희_ 중심선 구하기 _201020
for i in range(len(centerY)-1):
    # by 김주희_x값중 최대값과 최소값의 중간값 구하기-더해서 나누기2 _201020
    # 중심값의 x값의 차이가 20 이하이면 같은 배열에 대입한다.
    if abs(centerX[i+1] - centerX[i]) <= 20:
        temp.append(centerX[i])
    else:
        temp.append(centerX[i])
        sameLevelX.append(temp)
        midX.append(int((min(temp) + max(temp)) / 2))
        temp = []

    if abs(centerY[i+1] - centerY[i]) <= 20:
        temp2.append(centerY[i])
    else:
        temp2.append(centerY[i])
        sameLevelY.append(temp2)
        midY.append(int((min(temp2) + max(temp2)) / 2))
        temp2 = []

# by 김주희_수평선 그리기_201021
for i in range(len(midY)):
    plt.hlines(midY[i]-20, 0, 2592, colors='lightblue', linewidth=1)
    hline.append(midY[i]-20)
    if i == len(midY)-1:
        plt.hlines(midY[i]+20, 0, 2592, colors='lightblue', linewidth=1)
        hline.append(midY[i]+20)

# by 배아랑이_점자 영역 구분하는 수직선, 수평선_201023
hline.sort()

braille = []
midX.append(0)
midX.append(0)

# by 배아랑이_점자 영역 구분_201023
for i in range(len(midX)-2):
    if 75 <= (midX[i + 1] - midX[i]):
        if 120 <= (midX[i + 1] - midX[i]):
            if 150 <= (midX[i + 1] - midX[i]):
                # 1___1 -> 점자 2개
                if i != len(midX)-3:
                    braille.append([midX[i], midX[i]+40])
            else:
                # 1__1 -> 점자 2개  -> 2가지 1 _ _1 / 1_ _ 1
                if 75 <= (midX[i+2] - midX[i+1]):
                    if 120 <= (midX[i+2] - midX[i+1]):
                        if 150 <= (midX[i+2] - midX[i+1]):     # 1__1___1
                            braille.append([midX[i], midX[i]+40])
                        else:                                       # 1__1__1
                            if i != 0:
                                if 75 <= (midX[i] - midX[i-1]):
                                    if 120 <= (midX[i+2] - midX[i+1]):
                                        braille.append([])
                                    else:
                                        braille.append([midX[i], midX[i]+40])
                                else:
                                    braille.append([midX[i-1], midX[i]])
                    else:                                           # 1__1_1
                        if 75 <= (midX[i] - midX[i-1]):        # _1__1_1
                            braille.append([midX[i]-40, midX[i]])
                        else:                                       # 11__1_1
                            braille.append([midX[i-1], midX[i]])
                else:                                               # 1__11
                    braille.append([midX[i], midX[i]+40])
        else:
            # 1_1 -> 점자 2개
            if i != len(midX)-3:
                if 75 <= (midX[i] - midX[i-1]):             # _1_1
                    braille.append([midX[i]-40, midX[i]])
                else:                                           # 11_1
                    braille.append([midX[i-1], midX[i]])
            else:                                               # first
                braille.append([midX[i]-40, midX[i]])
    else:
        # 11 -> 점자 하나
        if i == len(midX)-3:
            braille.append([midX[i-1], midX[i]])


# by 김주희_6등분 선중 수직선 좌표과 저장되어 있응 braille을 이용하여 수직선 그리기 & vline 리스트에 선 좌표 넣기_201024
for i in range(len(braille)):
    plt.vlines(braille[i][0]-20, 0, 1944, colors='lightblue', linewidth=1)
    plt.vlines(braille[i][1]-20, 0, 1944, colors='lightblue', linewidth=1)
    plt.vlines(braille[i][1]+20, 0, 1944, colors='lightblue', linewidth=1)
    vline.append(braille[i][0]-20)
    vline.append(braille[i][1]-20)
    vline.append(braille[i][1]+20)



vline.append(max(vline)+40)
vline.append(max(vline)+40)
vline.append(max(vline)+40)

rotateCenter.sort()
binaryCode = []   # 점자를 2진수로 변환하여 저장
b = []
a= 0


# by 김주희_점자를 binary code로 변환하기 위해 중심점의 위치를 인덱스 형식으로 b라는 배열에 저장_201024
j=0
for point in rotateCenter:
    if (point[0] < max(vline)):
        if (point[0] > vline[(a+1)*3]) :
            if point != rotateCenter[-1]:
                a = a + 1
                j = (a)*3
                binaryCode.append(b)
                b = []

    if (vline[j] <= point[0] <= vline[j + 1]) and (hline[0] <= point[1] <= hline[1]):
        b.append(0)
    elif (vline[j] <= point[0] <= vline[j + 1]) and (hline[1] <= point[1] <= hline[2]):
        b.append(1)
    elif (vline[j] <= point[0] <= vline[j + 1]) and (hline[2] <= point[1] <= hline[3]):
        b.append(2)
    elif (vline[j + 1] <= point[0] <= vline[j + 2]) and (hline[0] <= point[1] <= hline[1]):
        b.append(3)
    elif (vline[j + 1] <= point[0] <= vline[j + 2]) and (hline[1] <= point[1] <= hline[2]):
        b.append(4)
    elif (vline[j + 1] <= point[0] <= vline[j + 2]) and (hline[2] <= point[1] <= hline[3]):
        b.append(5)
binaryCode.append(b)

# 숫자형 리스트를 문자열로 바꿔 저장하는 변수
korStr = []

#by 김주희_점자의 점 위치를 파악한 배열을 이용하여 바이너라 형식으로 변환_201024
korBinary = np.zeros((len(binaryCode), 6),np.int32)
for i in range(len(binaryCode)):
    for j in range(len(binaryCode[i])):
        korBinary[i][binaryCode[i][j]] = 1
    a = ''.join(map(str, korBinary[i]))
    korStr.append(a)

# dictionary에서 value와 key 바꾸기
table_kor_v = {v: k for k, v in table_kor.items()}

# by 김주희_table_kor_v에서 매칭시킨 한글을 순서대로 저장하는 리스트_201101
string =[]

# by 김주희_인덱스로 접근하기 위해 리스트 형식으로 만든다._201101
charList = list(table_kor_v.values())

# by 김주희_각 글자의 인덱스만 저장하는 리스트 _201101
charIdx = []

for i in range(len(korStr)):
    s = table_kor_v.get(korStr[i])
    string.append(s)
    charIdx.append(charList.index(s))


# by 김주희_점자 자음과 모음 맞춰서 배열 만들기_201102
# charIdx와 string의 인덱스가 다르므로 구분해야함.
# string에는 새로운 문자가 들어가지만 lndex_char에는 들어가지 않음.
# charIdx에도 string처럼 값을 넣어줘야 한다.

# j 는 charIdx와 string의 인덱스로 쓰이고
# m 은 for문 반복 횟수
j = 0
for m in range(len(string)-1):
    # by 김주희_한글자씩 나누기 _201101
    # by 김주희_ 자음_받침 일 경우 -> 자음 ㅏ 받침 _201101
    if (0 <= charIdx[j] <= 12) and (13 <= charIdx[j+1] <= 26):
        string.insert(j+1,'ㅏ')
        charIdx.insert(j+1, charList.index('ㅏ'))
        j = j + 2

    # by 김주희_ 받침_모음 일 경우 -> 받침 ㅇ 모음 _201101
    elif (13 <= charIdx[j] <= 26) and (27 <= charIdx[j+1] <= 42):
        string.insert(j+1,'ㅇ')
        charIdx.insert(j+1, -1)
        j = j + 2

    # by 김주희_ 받침/모음_줄임말 일 경우 -> 받침/모음 ㅇ 줄임말 _201102
    elif ((27 <= charIdx[j] <= 42) and (45 <= charIdx[j+1] <= 58)) or ((13 <= charIdx[j] <= 26) and (45 <= charIdx[j+1] <= 58)):
        string.insert(j+1,'ㅇ')
        charIdx.insert(j+1, -1)
        j = j + 2

    # by 김주희_ 받침/모음/줄임말_모음 일 경우 -> 받침/모음/줄임말 ㅇ 모음 _201102
    elif ((27 <= charIdx[j] <= 42) and (27 <= charIdx[j+1] <= 42)) or ((13 <= charIdx[j] <= 26) and (27 <= charIdx[j+1] <= 42)):
        string.insert(j + 1, 'ㅇ')
        charIdx.insert(j + 1, -1)
        j = j + 2
    elif ((45 <= charIdx[j] <= 58) and (27 <= charIdx[j+1] <= 42)):
        string.insert(j + 1, 'ㅇ')
        charIdx.insert(j + 1, -1)
        j = j + 2
    else:
        j = j+1

print(join_jamos( ''.join(string)))

# #
# plt.subplot(222)
# plt.subplot(122)
# # plt.imshow(equ, cmap='gray')
# # plt.title('histogram equalization')
# plt.imshow(imgClosing)
# plt.title('imgClosing')
# plt.axis('off')

# plt.hlines(center[0][1], 0, 2592, colors='pink', linewidth=1)

# plt.subplot(223)
# plt.imshow(imgRotate2, cmap='gray')
# plt.title('roatate 90')
# plt.axis('off')

#
# plt.imshow(imgClosing )
# plt.title('imgClosing image')
# plt.axis('off')

#
# #
# plt.subplot(224)

# plt.imshow(imgRotate3, cmap='gray')
# plt.title('roatate M3')
# plt.axis('off')


# plt.hlines(center[0][1], 0, 2592, colors='pink', linewidth=1)
# # # plt.hist()
# plt.imshow(img, cmap='gray')
# plt.title('original image')
# plt.axis('off')
plt.imshow(imgRotate)
plt.title('imgRotate Image')
plt.axis('off')
plt.show()