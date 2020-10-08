import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from collections import Counter
import statistics as st


# 카메라로 사진을 다시 찍어서 실행해봄. 명암이 균일하지 않아서 검출에 힘들었던 사진말고 조금더 명암이 균일한 조건에서 촬영함
# 조건 : 표지판과 카메라 10cm 거리, 카메라와 표지판이 평행하게 촬영


# img = cv.imread('./data/example.png', cv.IMREAD_GRAYSCALE)
# img = cv.imread('./data/example_1.jpeg', cv.IMREAD_GRAYSCALE)
# img = cv.imread('./data/smoking.png', cv.IMREAD_GRAYSCALE)
# img = cv.imread('./data/smoking_2.jpg', cv.IMREAD_GRAYSCALE)
img = cv.imread('./data/smoking_3.png', cv.IMREAD_GRAYSCALE)
img = cv.imread('./data/g.jpg', cv.IMREAD_GRAYSCALE)

# img_2 = cv.imread('./data/smoking.png')
# img_2 = cv.imread('./data/smoking_2.jpg')
# img_2 = cv.imread('./data/smoking_3.png')
img_2 = cv.imread('./data/g.jpg')

# 라즈베리파이로 사진 찍었을 때 180도 회전해서 보이기 때문에 회전시킴.
# img = cv.rotate(img, cv.ROTATE_180)
# img_2 = cv.rotate(img_2, cv.ROTATE_180)
thresh1 = img_2.copy()

# thr2가 더 선명하다.
thr1 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
thr2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)

thr3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)


# equ = cv.equalizeHist(thr2)
equ = cv.equalizeHist(img)

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
kernel = np.ones((1, 1), np.uint8)
# closing = cv.morphologyEx(thresh1, cv.MORPH_CLOSE, kernel)

closing = cv.morphologyEx(thresh1, cv.MORPH_CLOSE, kernel)
contours, _ = cv.findContours(closing[:, :, 0], cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
closing_2 = closing.copy()

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
    if (aspect_ratio > 0.4) and (aspect_ratio < 1.2):
        if (w>=5) and (h>=5):
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
cr = closing_2
cr = closing_2[:, :, 0]
# # cr[:, :, 1] = closing
# # cr[:, :, 2] = closing

# circles = cv.HoughCircles(cr,cv.HOUGH_GRADIENT,1,5,param1=50,param2=35,minRadius=1,maxRadius=0)
# circles = cv.HoughCircles(cr,cv.HOUGH_GRADIENT,1, 5)
circles = cv.HoughCircles(cr,cv.HOUGH_GRADIENT,1,100,param1=10,param2=35)

print(circles)
for c in circles[0, :]:
    center = (c[0], c[1])
    radius = c[2]

    # 바깥원
    cv.circle(closing_2, center, radius, (0, 255, 0), 2)

    # 중심원
    cv.circle(closing_2, center, 2, (0, 0, 255), 3)


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
# plt.hist()
plt.imshow(closing_2, cmap='gray')
plt.title('closing_2 image')
plt.axis('off')

plt.show()