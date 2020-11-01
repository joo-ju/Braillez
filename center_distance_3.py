import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from collections import Counter
import statistics as st


# img = cv.imread('./data/example.png', cv.IMREAD_GRAYSCALE)
# img = cv.imread('./data/example_1.jpeg', cv.IMREAD_GRAYSCALE)
# img = cv.imread('./data/smoking.png', cv.IMREAD_GRAYSCALE)
# img = cv.imread('./data/smoking_2.jpg', cv.IMREAD_GRAYSCALE)
# img = cv.imread('./data/smoking_3.png', cv.IMREAD_GRAYSCALE)
img = cv.imread('./data/n.jpg', cv.IMREAD_GRAYSCALE)

# img_2 = cv.imread('./data/smoking.png')
# img_2 = cv.imread('./data/smoking_2.jpg')
# img_2 = cv.imread('./data/smoking_3.png')
img_2 = cv.imread('./data/n.jpg')

img = img / 255
img_2 = img_2 / 255

print(img)
img_180 = cv.rotate(img, cv.ROTATE_180)
# thresh1 = img_2
thresh1 = img_180

# thr2가 더 선명하다.
# 이미지의 명암이 일정하지 않기 때문에 가변적으로 threshold를 한다.
# thr1 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
thr2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
# thr2 = cv.adaptiveThreshold(img_180, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
#
# thr3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)


# equ = cv.equalizeHist(thr2)
# equ = cv.equalizeHist(img)

ret,thr = cv.threshold(equ,210,255,cv.THRESH_BINARY)

# thresh1[:, :, 0] = thr
# thresh1[:, :, 1] = thr
# thresh1[:, :, 2] = thr

thresh1[:, :, 0] = thr2
thresh1[:, :, 1] = thr2
thresh1[:, :, 2] = thr2
# hist = cv.calcHist()
# thr2 = 255 - thr2

# cv.drawContours(img_thr, contours, -1, (0, 255, 0), 1)

compare_area=[]

rect_center = []


center_point = []
center_sort = []
dst = []

box = []
#


# kernel을 (3, 3)로 하면 완전히 적게 제거됨
kernel = np.ones((3, 3), np.uint8)
# closing = cv.morphologyEx(thresh1, cv.MORPH_CLOSE, kernel)

closing = cv.morphologyEx(thresh1, cv.MORPH_CLOSE, kernel)
contours, _ = cv.findContours(closing[:, :, 0], cv.RETR_TREE, cv.CHAIN_APPROX_NONE)


for i in range(len(contours)):
    cnt = contours[i]
    area = cv.contourArea(cnt)
    x, y, w, h = cv.boundingRect(cnt)


    # by 김주희_contour 면적 _200702
    rect_area = w * h
    compare_area.append(rect_area)

    # by 김주희_가로 세로의 비율 _200702
    aspect_ratio = float(w)/h


    # by 김주희_컨투어한 영역의 비율을 보고 사각형을 그림 _200702
    #   점자에 대한 contour를 찾는 과정_가로 세로 비율이 1:1에서 크게 벗어난 것을 제외하고 표시
    # if(aspect_ratio>0.8)and(aspect_ratio<1.5)and(f <= float(f)*0.7):

    # cv.rectangle(closing, (x, y), (x + w, y + h), (0, 127, 127), 10)
    if (aspect_ratio > 0.4) and (aspect_ratio < 1.0):
        box.append(cv.boundingRect(cnt))
        cv.rectangle(closing, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # by 김주희_moments를 이용하여 중심점 표_200702
        # mnt = cv.moments(cnt)
        # if int(mnt['m00']) != 0:
        #     cx = int(mnt['m10'] / mnt['m00'])
        #     cy = int(mnt['m01'] / mnt['m00'])
        #     cv.circle(thresh1, (cx, cy), 2, (255, 0, 0), -1)
            # by_김주희_기울기 및 거리 구하기위해 중심 값 배열로 만듦 _200819
            # center_point.append([cx,cy])
            # center_y.append(cy)
            # rect_center = rect_center + [cx, cy]

plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('image gray')
plt.axis('off')
#
plt.subplot(222)
# plt.imshow(equ, cmap='gray')
# plt.title('histogram equalization')
plt.imshow(thr2, cmap='gray')
plt.title('thr2')
plt.axis('off')
#
plt.subplot(223)
# plt.imshow(thr, cmap='gray')
# plt.title('Binary image')
# plt.axis('off')


plt.imshow(closing )
plt.title('closing image')
plt.axis('off')



plt.subplot(224)
# plt.hist(equ)
plt.imshow(img_180, cmap='gray')
plt.title('rotate image')
plt.axis('off')

plt.show()

