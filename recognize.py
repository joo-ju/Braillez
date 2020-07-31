
# 노이즈 제거

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from collections import Counter
import statistics as st


# img = cv.imread('./data/example.png')
img = cv.imread('./data/example_1.jpeg')
img_thr = img.copy()

# by 김주희_그레이스케일 영상으로 변_200701
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# by 김주희_임계값을 지정하여 binary image(이진화 영상)으로 변환 _200701
ret, thr = cv.threshold(imgray, 210, 255, cv.THRESH_BINARY)
# thr = cv.adaptiveThreshold(imgray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, 20)

# by 김주희_이진화 흑백 반전_200701
thr = 255-thr

# by 김주희_이진화 영상을 3채널로 변경_200701
img_thr[:, :, 0] = thr
img_thr[:, :, 1] = thr
img_thr[:, :, 2] = thr


# by 김주희_morphologyEx 함수를 이용하여 점자외의 점들을 제거하기 _200708
# 가우시안 블러링 잘 안됨 -> 제거되지 않음
# kernel을 (5, 5)로 하면 완전히 적게 제거됨
# kernel = cv.getGaussianKernel(5, 0.1)
# img_gaussian = cv.filter2D(img_thr, -1, kernel)
kernel = np.ones((3, 3), np.uint8)
closing = cv.morphologyEx(img_thr, cv.MORPH_CLOSE, kernel)



# by 김주희_Contour 함수를 통해 점자 영역 표시_200701
# contours, _ = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
contours, _ = cv.findContours(closing[:, :, 0], cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

# cv.drawContours(img_thr, contours, -1, (0, 255, 0), 1)

# by 김주희_contour영역의 넓이 비교를 위한 변수_200702
compare_area=[]

# by 김주희_검출된 contour에 대해 점자인 것과 아닌 것을 구분하여 표시 _200702
for i in range(len(contours)):
    cnt = contours[i]
    area = cv.contourArea(cnt)
    x, y, w, h = cv.boundingRect(cnt)

    # by 김주희_contour 면적_200702
    rect_area = w * h
    compare_area.append(rect_area)

    # by 김주희_가로 세로의 비율_200702
    aspect_ratio = float(w)/h

    # by 김주희_컨투어 넓이의 최빈값 구하기 _200702
    c = Counter(compare_area)
    frequency = c.most_common(1)
    f = frequency[0][0]

    print(aspect_ratio)
    print("compare_area : ", compare_area)
    print("frequency : ", f)

    # by 김주희_컨투어한 영역의 비율을 보고 사각형을 그림_200702
    #   점자에 대한 contour를 찾는 과정_가로 세로 비율이 1:1에서 크게 벗어난 것을 제외하고 표시
    # if(aspect_ratio>0.8)and(aspect_ratio<1.5)and(f <= float(f)*0.7):
    if (aspect_ratio > 0.5) and (aspect_ratio < 2.0):
        cv.rectangle(closing, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # by 김주희_moments를 이용하여 중심점 표_200702
        mnt = cv.moments(cnt)
        if int(mnt['m00']) != 0:
            cx = int(mnt['m10'] / mnt['m00'])
            cy = int(mnt['m01'] / mnt['m00'])
            cv.circle(closing, (cx, cy), 2, (255, 0, 0), -1)



# plt.imshow(img_thr, cmap='gray')
plt.subplot(1, 2, 1)
# plt.imshow(img_thr, cmap='gray')
plt.imshow(closing, cmap='gray')
plt.title('closing')
plt.axis('off')
#
plt.subplot(1, 2, 2)
plt.imshow(imgray, cmap='gray')
# plt.imshow(img_gaussian, cmap='gray')
plt.title('img_thr')
plt.axis('off')

plt.show()




