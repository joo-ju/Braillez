import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
from collections import Counter

# 카메라로 사진을 다시 찍어서 실행해봄. 명암이 균일하지 않아서 검출에 힘들었던 사진말고 조금더 명암이 균일한 조건에서 촬영함
# 조건 : 표지판과 카메라 10cm 거리, 카메라와 표지판이 평행하게 촬영


# img = cv.imread('./data/example.png', cv.IMREAD_GRAYSCALE)
# img = cv.imread('./data/example_1.jpeg', cv.IMREAD_GRAYSCALE)
# img = cv.imread('./data/smoking.png', cv.IMREAD_GRAYSCALE)
# img = cv.imread('./data/smoking_2.jpg', cv.IMREAD_GRAYSCALE)
# img = cv.imread('./data/smoking_3.png', cv.IMREAD_GRAYSCALE)
# img = cv.imread('./data/g.jpg', cv.IMREAD_GRAYSCALE)
img = cv.imread('./data/f.jpg', cv.IMREAD_GRAYSCALE)
# img = cv.imread('./data/e.jpg', cv.IMREAD_GRAYSCALE)

# img_2 = cv.imread('./data/smoking.png')
# img_2 = cv.imread('./data/smoking_2.jpg')
# img_2 = cv.imread('./data/smoking_3.png')
# img_2 = cv.imread('./data/g.jpg')
img_2 = cv.imread('./data/f.jpg')
# img_2 = cv.imread('./data/e.jpg')

# 컨투어 영역의 넓이 중 최빈값으 구하기 위한 배
compare_area = []

yuv = cv.cvtColor(img_2, cv.COLOR_BGR2YUV)

y = yuv[:, :, 0]

sobel = cv.Sobel(img_2, cv.CV_8U, 1, 0, 3)
laplacian = cv.Laplacian(img_2, cv.CV_8U, ksize=3)
canny = cv.Canny(img_2, 10, 255)

# sobel 이진 영상으로 변환
# 반전(검->흰, 흰->검)
sobel = 255 - sobel

y = 255 - y

y2 = yuv

y2[:, :, 0] = yuv[:, :, 0]
y2[:, :, 1] = yuv[:, :, 0]
y2[:, :, 2] = yuv[:, :, 0]
sobel = cv.Sobel(y2, cv.CV_8U, 1, 0, 3)
#201007_이진 영상에서 점자 검출

ret,thr = cv.threshold(y2,210,255,cv.THRESH_BINARY)
print(thr.shape)
#
# y2[:, :, 0] = thr
# y2[:, :, 1] = thr
# y2[:, :, 2] = thr
#
# thresh1[:, :, 0] = thr2
# thresh1[:, :, 1] = thr2
# thresh1[:, :, 2] = thr2


kernel = np.ones((3, 3), np.uint8)
closing = cv.morphologyEx(thr, cv.MORPH_CLOSE, kernel)

# by 배아랑이_201012
# 이미지의 가장자리 일정 부분은 하얗게 만든다. -> 그림 잘라 버리기
# 그림의 가로세로 길이의 8분의 1 정도만 가장자리를 검게 바꿔준다.
cut_num = [(int)(closing.shape[0]/8), (int)(closing.shape[1]/8)]
print(cut_num)
closing[:cut_num[0], :] = 125
closing[-cut_num[0]:, :] = 125
closing[:, :cut_num[1]] = 125
closing[:, -cut_num[1]:] = 125


# closing = cv.morphologyEx(y2, cv.MORPH_CLOSE, kernel)
contours, _ = cv.findContours(closing[:, :, 0], cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
center=[]
for i in range(len(contours)):
    cnt = contours[i]
    area = cv.contourArea(cnt)
    x, y, w, h = cv.boundingRect(cnt)


    # by 김주희_contour 면적 _200702
    rect_area = w * h
    compare_area.append(rect_area)


    # 컨투어 영역이 1인것 없애보기 -임시
    # if rect_area == 1:
        # closing[x, y, 0]

    # by 김주희_가로 세로의 비율 _200702
    aspect_ratio = float(w)/h


    # by 김주희_컨투어한 영역의 비율을 보고 사각형을 그림 _200702
    #   점자에 대한 contour를 찾는 과정_가로 세로 비율이 1:1에서 크게 벗어난 것을 제외하고 표시
    # if(aspect_ratio>0.8)and(aspect_ratio<1.5)and(f <= float(f)*0.7):

    # cv.rectangle(closing, (x, y), (x + w, y + h), (0, 127, 127), 10)
    if (aspect_ratio > 0.4) and (aspect_ratio < 1.2):
        # 크기가 10픽셀 이하인 것은 모두 없애기._201012
        if (w > 10) :
            # cv.rectangle(closing, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cx = int(x + w/2)
            cy = int(y + h/2)
            mnt = cv.moments(cnt)

            # by 김주희_컨투어한 영역의 중심점이 아닌 사각형의 중심점을 표시함._201012
            cv.circle(closing, (cx, cy), 3, (255, 0, 0), -1)
            center.append([cx, cy])



print('center', center)
print('center size', len(center))


# by 김주희_점자 중 가장 처음의 2점 혹은 가장 마지막의 2점에 대한 기울기 _201012
dx = center[-2][0] - center[-1][0]
dy = center[-2][1] - center[-1][1]

dx2 = center[0][0] - center[2][0]
dy2 = center[0][1] - center[2][1]

print('dx', dx)
print('dy', dy)
print('dx2', dx2)
print('dy2', dy2)

# by 김주희_두 기울기의 평균으로 회전하기_201019
ave = (math.atan(dy/dx) + math.atan(dy2/dx2)) / 2
# math.atan(dy2/dx2)
print('atan', ave)

# by 김주희_ 각도 구하기 _201012
angle = ave * (180.0 / math.pi)

h, w = closing.shape[:2]
# M1 = cv.getRotationMatrix2D((w/2, h/2), 10, 1)
M = cv.getRotationMatrix2D((w/2, h/2), angle, 1)
rotation = cv.warpAffine(closing, M,(w, h))

# by 김주희_ 현재 closing에 contour 실시함 -> 사각형 그려져 있음._201012
# by 김주희_ 거리를 계산하여 주변에 점이 많으면 같이 없앰.
# 넓이들중에 최빈 값을 구한다.
cnt = Counter(compare_area)
# print(cnt.most_common(3))
# print(cnt)
#
# print(closing.shape)
ro_center = []
center_x = []
center_y = []
contours, _ = cv.findContours(rotation[:, :, 0], cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
for i in range(len(contours)):
    cnt = contours[i]
    area = cv.contourArea(cnt)
    x, y, w, h = cv.boundingRect(cnt)


    # by 김주희_contour 면적 _200702
    rect_area = w * h
    compare_area.append(rect_area)


    # 컨투어 영역이 1인것 없애보기 -임시
    # if rect_area == 1:
        # closing[x, y, 0]

    # by 김주희_가로 세로의 비율 _200702
    aspect_ratio = float(w)/h


    # by 김주희_컨투어한 영역의 비율을 보고 사각형을 그림 _200702
    #   점자에 대한 contour를 찾는 과정_가로 세로 비율이 1:1에서 크게 벗어난 것을 제외하고 표시
    # if(aspect_ratio>0.8)and(aspect_ratio<1.5)and(f <= float(f)*0.7):

    # cv.rectangle(closing, (x, y), (x + w, y + h), (0, 127, 127), 10)
    if (aspect_ratio > 0.4) and (aspect_ratio < 1.2):
        # 크기가 10픽셀 이하인 것은 모두 없애기._201012
        if (w > 10) :
            cv.rectangle(rotation, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cx = int(x + w/2)
            cy = int(y + h/2)
            mnt = cv.moments(cnt)

            # by 김주희_컨투어한 영역의 중심점이 아닌 사각형의 중심점을 표시함._201012
            cv.circle(rotation, (cx, cy), 3, (255, 0, 0), -1)
            ro_center.append([cx, cy])
            center_x.append(cx)
            center_y.append(cy)




#by 김주희_ _201019
print("변경된 ro_center : ", ro_center)

print("변경된 center_y : ", center_y)
print("변경된 center_x : ", center_x)
center_x.sort()
print("정렬된 center_x : ", center_x)
diff = 0
count = 0
sum = 0
count_x = 0
sum_x = 0
hline = []
vline = []

# by 김주희_ 거리를 구할 때 인덱스오류가 나서 하나 더 대입해줌 - 영향 없음_201019
center_y.append(0)
center_x.append(0)
center_y.insert(0, 0)
center_x.insert(0, 0)

print("추가한 center_x : ", center_x)
# by 김주희_y값 기준 만약에 center의 거리가 30px이하이면 카운트 해주고, 값을 더해준다. (나중에 평균 구할 예정) _201019
for i in range(len(center_y)-1):
    # by 김주희_x 값 평균 구하기_201019
    if abs(center_x[i+1] - center_x[i]) >= 20:
        if (i != 0) :
            count_x = count_x + 1
            sum_x = sum_x + center_x[i]
            vline.append(int(sum_x/count_x))
        count_x = 0
        sum_x = 0
    else:
        count_x = count_x + 1
        sum_x = sum_x + center_x[i]
        print("sum_x : ", sum_x)
        print("count_x : ", count_x)

    # by 김주희_y 값 평균 구하기_201019
    if abs(center_y[i+1] - center_y[i]) >= 30:
        if (i != 0) :
            count = count + 1
            sum = sum + center_y[i]
            hline.append(int(sum/count))
        count = 0
        sum = 0
    else:
        count = count + 1
        sum = sum + center_y[i]
        print("sum : ", sum)
        print("count : ", count)

print("hline", hline)
print("vline", vline)




#by 김주희_점에서 가장 가까운 수 찾기 _201019
center_arr = np.asarray(a=ro_center)
arr_center_y = np.asarray(a=center_y)
value = center_y[0]
print("arr_center_y.shape ", arr_center_y.shape)
# result = np.find_nearest(arr_center_y, value)
# print("find_nearest ( center_y[0] :", center_y[0], ") : ",result )
# print("center_arr : ", center_arr)

#by 김주희_ _201019
# #by 김주희_ _201019



# plt.subplot(221)
plt.subplot(121)
plt.imshow(rotation, cmap='gray')
plt.title('roatate')
plt.axis('off')

# by 김주희_ 중심점에 맞는 수평선 그리기 _201012
# plt.hlines(center[0][1], 0, 2592, colors='pink', linewidth=1)

# by 김주희_ 근접한 y값들의 평균을 구해서 수직선 그리기._201019
for i in range(len(hline)):
    plt.hlines(hline[i], 0, 2592, colors='pink', linewidth=1)

for i in range(len(vline)):
    plt.vlines(vline[i], 0, 1944, colors='pink', linewidth=1)
# #
# plt.subplot(222)
plt.subplot(122)
# plt.imshow(equ, cmap='gray')
# plt.title('histogram equalization')
plt.imshow(closing)
plt.title('closing')
plt.axis('off')

plt.hlines(center[0][1], 0, 2592, colors='pink', linewidth=1)

# plt.subplot(223)
# plt.imshow(rotation2, cmap='gray')
# plt.title('roatate 90')
# plt.axis('off')

#
# plt.imshow(closing )
# plt.title('closing image')
# plt.axis('off')

#
# #
# plt.subplot(224)

# plt.imshow(rotation3, cmap='gray')
# plt.title('roatate M3')
# plt.axis('off')


# plt.hlines(center[0][1], 0, 2592, colors='pink', linewidth=1)
# # # plt.hist()
# plt.imshow(img, cmap='gray')
# plt.title('original image')
# plt.axis('off')

plt.show()