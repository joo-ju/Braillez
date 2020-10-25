import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
from collections import Counter
from braille import table_kor
from unicode import join_jamos




img = cv.imread('./data/o.jpg')
img_2 = cv.imread('./data/o.jpg')

img = cv.resize(img, (2592, 1944))
img_2 = cv.resize(img_2, (2592, 1944))


yuv = cv.cvtColor(img_2, cv.COLOR_BGR2YUV)

y = yuv[:, :, 0]


y = 255 - y

y2 = yuv

y2[:, :, 0] = yuv[:, :, 0]
y2[:, :, 1] = yuv[:, :, 0]
y2[:, :, 2] = yuv[:, :, 0]


ret,thr = cv.threshold(y2,210,255,cv.THRESH_BINARY)



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


plt.subplot(121)

plt.imshow(img, cmap='gray')
plt.title('img')
plt.axis('off')


plt.subplot(122)
# plt.hlines(center[0][1], 0, 2592, colors='pink', linewidth=1)
# # plt.hist()
plt.imshow(y2, cmap='gray')
plt.title(' y')
plt.axis('off')

plt.show()