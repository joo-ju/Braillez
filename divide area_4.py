import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
from collections import Counter
from braille import table_kor


img = cv.imread('./data/n.jpg', cv.IMREAD_GRAYSCALE)
# img = cv.imread('./data/n.jpg')
img = cv.rotate(img, cv.ROTATE_180)
img_2 = cv.imread('./data/n.jpg')
img_2 = cv.rotate(img_2, cv.ROTATE_180)

img = cv.resize(img, (2592, 1944))
img_2 = cv.resize(img_2, (2592, 1944))


compare_area = []

yuv = cv.cvtColor(img_2, cv.COLOR_BGR2YUV)

y = yuv[:, :, 0]

sobel = cv.Sobel(img_2, cv.CV_8U, 1, 0, 3)

sobel = 255 - sobel

y = 255 - y

y2 = yuv

y2[:, :, 0] = yuv[:, :, 0]
y2[:, :, 1] = yuv[:, :, 0]
y2[:, :, 2] = yuv[:, :, 0]
sobel = cv.Sobel(y2, cv.CV_8U, 1, 0, 3)

ret,thr = cv.threshold(y2,210,255,cv.THRESH_BINARY)

kernel = np.ones((3, 3), np.uint8)
closing = cv.morphologyEx(thr, cv.MORPH_CLOSE, kernel)

cut_num = [(int)(closing.shape[0]/8), (int)(closing.shape[1]/8)]
closing[:cut_num[0], :] = 125
closing[-cut_num[0]:, :] = 125
closing[:, :cut_num[1]] = 125
closing[:, -cut_num[1]:] = 125

# 추가
closing1 = np.array(closing[:, :, 0], dtype=np.uint8)
# closing1 = 255-closing1
contours, _ = cv.findContours(closing1, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

center=[]
for i in range(len(contours)):
    cnt = contours[i]
    area = cv.contourArea(cnt)
    x, y, w, h = cv.boundingRect(cnt)


    rect_area = w * h
    compare_area.append(rect_area)
    aspect_ratio = float(w)/h

    if (aspect_ratio > 0.8) and (aspect_ratio < 1.2):
        if (w > 5) and (w < 25):
            cv.rectangle(closing, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cx = int(x + w/2)
            cy = int(y + h/2)
            mnt = cv.moments(cnt)

            cv.circle(closing, (cx, cy), 3, (255, 0, 0), -1)
            center.append([cx, cy])



dx = center[-2][0] - center[-1][0]
dy = center[-2][1] - center[-1][1]

dx2 = center[0][0] - center[5][0]
dy2 = center[0][1] - center[5][1]


ave = math.atan(dy2/dx2)

angle = ave * (180.0 / math.pi)

h, w = closing.shape[:2]
M = cv.getRotationMatrix2D((w/2, h/2), angle, 1)
rotation = cv.warpAffine(closing, M,(w, h))
cnt = Counter(compare_area)
ro_center = []
center_x = []
center_y = []
contours, A = cv.findContours(rotation[:, :, 0], cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
for i in range(len(contours)):
    cnt = contours[i]
    area = cv.contourArea(cnt)
    x, y, w, h = cv.boundingRect(cnt)


    rect_area = w * h
    compare_area.append(rect_area)

    aspect_ratio = float(w)/h

    if (aspect_ratio > 0.8) and (aspect_ratio < 1.2):
        if (w > 5) and (w < 25):
            cv.rectangle(rotation, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cx = int(x + w/2)
            cy = int(y + h/2)
            mnt = cv.moments(cnt)

            cv.circle(rotation, (cx, cy), 3, (255, 0, 0), -1)
            ro_center.append([cx, cy])
            center_x.append(cx)
            center_y.append(cy)



center_x.sort()
center_y.sort()
diff = 0
count = 0
count_x = 0
hline = []
vline = []

sameLevel_y = []
sameLevel_x = []
temp = []
temp2 = []
mid_x = []
mid_y = []

center_y.append(0)
center_x.append(0)


for i in range(len(center_y)-1):

    if abs(center_x[i+1] - center_x[i]) <= 20:
        temp.append(center_x[i])
    else:
        temp.append(center_x[i])
        sameLevel_x.append(temp)
        mid_x.append(int((min(temp) + max(temp)) / 2))
        temp = []


    if abs(center_y[i+1] - center_y[i]) <= 20:
        temp2.append(center_y[i])
    else:
        temp2.append(center_y[i])
        sameLevel_y.append(temp2)
        mid_y.append(int((min(temp2) + max(temp2)) / 2))
        temp2 = []

for i in range(len(mid_y)):
    plt.hlines(mid_y[i]-20, 0, 2592, colors='lightblue', linewidth=1)
    hline.append(mid_y[i]-20)
    if i == len(mid_y)-1:
        plt.hlines(mid_y[i]+20, 0, 2592, colors='lightblue', linewidth=1)
        hline.append(mid_y[i]+20)
hline.sort()

braille = []
count = 0
mid_x.append(0)
mid_x.append(0)
for i in range(len(mid_x)-2):
    if 75 <= (mid_x[i + 1] - mid_x[i]):
        if 120 <= (mid_x[i + 1] - mid_x[i]):
            if 150 <= (mid_x[i + 1] - mid_x[i]):
                if i != len(mid_x)-3:
                    braille.append([mid_x[i], mid_x[i]+40])
            else:
                if 75 <= (mid_x[i+2] - mid_x[i+1]):
                    if 120 <= (mid_x[i+2] - mid_x[i+1]):
                        if 150 <= (mid_x[i+2] - mid_x[i+1]):     # 1__1___1
                            braille.append([mid_x[i], mid_x[i]+40])
                        else:                                       # 1__1__1
                            if i != 0:
                                if 75 <= (mid_x[i] - mid_x[i-1]):
                                    if 120 <= (mid_x[i+2] - mid_x[i+1]):
                                        braille.append([])
                                    else:
                                        braille.append([mid_x[i], mid_x[i]+40])
                                else:
                                    braille.append([mid_x[i-1], mid_x[i]])
                    else:                                           # 1__1_1
                        if 75 <= (mid_x[i] - mid_x[i-1]):        # _1__1_1
                            braille.append([mid_x[i]-40, mid_x[i]])
                        else:                                       # 11__1_1
                            braille.append([mid_x[i-1], mid_x[i]])
                else:                                               # 1__11
                    braille.append([mid_x[i], mid_x[i]+40])
        else:
            # 1_1 -> 점자 2개
            if i != len(mid_x)-3:
                if 75 <= (mid_x[i] - mid_x[i-1]):             # _1_1
                    braille.append([mid_x[i]-40, mid_x[i]])
                else:                                           # 11_1
                    braille.append([mid_x[i-1], mid_x[i]])
            else:                                               # first
                braille.append([mid_x[i]-40, mid_x[i]])
    else:
        if i == len(mid_x)-3:
            braille.append([mid_x[i-1], mid_x[i]])


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
ro_center.sort()
binary_code = []   # 점자를 2진수로 변환하여 저장
b = []
a= 0

j=0
for point in ro_center:
    if (point[0] < max(vline)):
        if (point[0] > vline[(a+1)*3]) :
            if point != ro_center[-1]:
                a = a + 1
                j = (a)*3
                binary_code.append(b)
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


binary_code.append(b)
kor_d = []#


kor_b = np.zeros((len(binary_code), 6),np.int32)
for i in range(len(binary_code)):
    for j in range(len(binary_code[i])):
        kor_b[i][binary_code[i][j]] = 1
    a = ''.join(map(str, kor_b[i]))
    kor_d.append(a)


table_kor_v = {v: k for k, v in table_kor.items()}



if kor_d[0] == '000101':
    str = "장애인전용"
if kor_d[0] == '100011':
    str = "여자화장실"
