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

# closing = cv.morphologyEx(y2, cv.MORPH_CLOSE, kernel)
contours, _ = cv.findContours(closing[:, :, 0], cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
for i in range(len(contours)):
    cnt = contours[i]
    area = cv.contourArea(cnt)
    x, y, w, h = cv.boundingRect(cnt)


    # by 김주희_contour 면적 _200702
    rect_area = w * h
    compare_area.append(rect_area)

    # 컨투어 영역이 1인것 없애보기 -임시
    if rect_area == 1:
        closing[x, y, 0]

    # by 김주희_가로 세로의 비율 _200702
    aspect_ratio = float(w)/h


    # by 김주희_컨투어한 영역의 비율을 보고 사각형을 그림 _200702
    #   점자에 대한 contour를 찾는 과정_가로 세로 비율이 1:1에서 크게 벗어난 것을 제외하고 표시
    # if(aspect_ratio>0.8)and(aspect_ratio<1.5)and(f <= float(f)*0.7):

    # cv.rectangle(closing, (x, y), (x + w, y + h), (0, 127, 127), 10)
    if (aspect_ratio > 0.4) and (aspect_ratio < 1.2):

        # box.append(cv.boundingRect(cnt))
        cv.rectangle(closing, (x, y), (x+w, y+h), (0, 255, 0), 2)

# 현재 closing에 contour 실시함 -> 사각형 그려져 있음.
# 거리를 계산하여 주변에 점이 많으면 같이 없앰.
# 넓이들중에 최빈 값을 구한다.
cnt = Counter(compare_area)
print(cnt.most_common(3))
print(cnt)

print(closing.shape)

plt.subplot(221)
plt.imshow(yuv, cmap='gray')
plt.title('yuv')
plt.axis('off')
# #
plt.subplot(222)
# plt.imshow(equ, cmap='gray')
# plt.title('histogram equalization')
plt.imshow(closing)
plt.title('closing')
plt.axis('off')
# #
plt.subplot(223)
plt.imshow(y2, cmap='gray')
plt.title('YUV-Y')
plt.axis('off')

#
# plt.imshow(closing )
# plt.title('closing image')
# plt.axis('off')

#
#
plt.subplot(224)
# # plt.hist()
plt.imshow(img, cmap='gray')
plt.title('original image')
plt.axis('off')

plt.show()