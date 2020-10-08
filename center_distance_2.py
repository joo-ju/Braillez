import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from collections import Counter
import statistics as st


# img = cv.imread('./data/example.png')
# img = cv.imread('./data/example_1.jpeg')
img = cv.imread('./data/smoking.png')
img_thr = img.copy()

# by 김주희_그레이스케일 영상으로 변 _200701
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# by 김주희_임계값을 지정하여 binary image(이진화 영상)으로 변환 _200701
ret, thr = cv.threshold(imgray, 210, 255, cv.THRESH_BINARY)
# thr = cv.adaptiveThreshold(imgray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, 20)

# by 김주희_이진화 흑백 반전 _200701
thr = 255-thr

# by 김주희_이진화 영상을 3채널로 변경 _200701
img_thr[:, :, 0] = thr
img_thr[:, :, 1] = thr
img_thr[:, :, 2] = thr


# by 김주희_morphologyEx 함수를 이용하여 점자외의 점들을 제거하기 _200708
# 가우시안 블러링 잘 안됨 -> 제거되지 않음
# kernel = cv.getGaussianKernel(5, 0.1)
# img_gaussian = cv.filter2D(img_thr, -1, kernel)

# kernel을 (3, 3)로 하면 완전히 적게 제거됨
kernel = np.ones((3, 3), np.uint8)
closing = cv.morphologyEx(img_thr, cv.MORPH_CLOSE, kernel)



# by 김주희_Contour 함수를 통해 점자 영역 표시 _200701
# contours, _ = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
contours, _ = cv.findContours(closing[:, :, 0], cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

# cv.drawContours(img_thr, contours, -1, (0, 255, 0), 1)

# by 김주희_contour영역의 넓이 비교를 위한 변수 _200702
compare_area=[]

# by 배아랑이_중심점 저장을 위한 변수 _200708
rect_center = []


center_point = []
center_sort = []
dst = []

box = []

# by 김주희_검출된 contour에 대해 점자인 것과 아닌 것을 구분하여 표시 _200702
for i in range(len(contours)):
    cnt = contours[i]
    area = cv.contourArea(cnt)
    x, y, w, h = cv.boundingRect(cnt)


    # by 김주희_contour 면적 _200702
    rect_area = w * h
    compare_area.append(rect_area)

    # by 김주희_가로 세로의 비율 _200702
    aspect_ratio = float(w)/h

    # by 김주희_컨투어 넓이의 최빈값 구하기 _200702
    c = Counter(compare_area)
    frequency = c.most_common(1)
    f = frequency[0][0]

    # print(aspect_ratio)
    # print("compare_area : ", compare_area)
    # print("frequency : ", f)

    # by 김주희_컨투어한 영역의 비율을 보고 사각형을 그림 _200702
    #   점자에 대한 contour를 찾는 과정_가로 세로 비율이 1:1에서 크게 벗어난 것을 제외하고 표시
    # if(aspect_ratio>0.8)and(aspect_ratio<1.5)and(f <= float(f)*0.7):
    if (aspect_ratio > 0.5) and (aspect_ratio < 2.0):
        box.append(cv.boundingRect(cnt))
        cv.rectangle(closing, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # by 김주희_moments를 이용하여 중심점 표_200702
        mnt = cv.moments(cnt)
        if int(mnt['m00']) != 0:
            cx = int(mnt['m10'] / mnt['m00'])
            cy = int(mnt['m01'] / mnt['m00'])
            cv.circle(closing, (cx, cy), 2, (255, 0, 0), -1)
            # by_김주희_기울기 및 거리 구하기위해 중심 값 배열로 만듦 _200819
            center_point.append([cx,cy])
            # center_y.append(cy)
            rect_center = rect_center + [cx, cy]


print(box)
print(box[0][1])

box.sort()
print(box)
print(box[0][1])
# len(box)
count = [0 for i in range(len(box))]
print(count)
# count.shape()

#by_김주희__200904
start = 0
for i in range(len(box)-1):
    d = box[i + 1][0] - box[i][0]
    # 점의 x값의 위치와 가로의 길이모두 일정 범위 안에 있는지 점검
    if box[i][2]-7<box[i+1][2]<box[i][2]+7:
        if box[i][0]-5<box[i+1][0]<box[i][0]+5:
            count[start] = count[start] + 1.0
            cv.circle(closing, (center_point[i][0], center_point[i][1]), 5, (0, 255, 255), -1)
        elif 60<d<70 or 130<d<140:
            count[start] = count[start] + 1.0
            cv.circle(closing, (center_point[i][0], center_point[i][1]), 5, (0, 255, 255), -1)

    else:
        start = i
    #
    # # 점의 x값의 위치와 가로의 길이모두 일정 범위 안에 있는지 점검
    # if box[i][0]-5<box[i+1][0]<box[i][0]-5 and box[i][2]-5<box[i+1][2]<box[i][2]+5:
    #     count[start] = count[start] + 1.0
    #     cv.circle(closing, (center_point[i][0], center_point[i][1]), 5, (0, 255, 255), -1)
    # # x값의 차이가 바로 옆 점자 혹은 다음 글자인지 확인
    # elif 60<d<70 or 130<d<140:
    #     count[start] = count[start] + 1.0
    #     cv.circle(closing, (center_point[i][0], center_point[i][1]), 5, (0, 255, 255), -1)
    # else:
    #     start = i


print(count)


#
# for i in range(len(box)): ##Buble Sort on python
#     for j in range(len(box)-(i+1)):
#         if box[j][0]>box[j+1][0]:
#             temp=box[j]
#             box[j]=box[j+1]
#             box[j+1]=temp
# print(box)
# print(box[0][1])



# by_김주희_기울기 및 거리 구하기_200819
rect_center = np.array(rect_center)
# print(rect_center[0])
# cv.circle(closing, (center_x[0], center_y[0]), 10, (255, 255, 0), -1)

# for i in range(4):
#     if i%2 == 0:
#         print("x" , int(i/2+1) ," :" , rect_center[i] )
#     if i%2 == 1:
#         print("y",int(i/2+1) , ":" ,rect_center[i])
# # print(int(rect_center[0]))
#
# # by_김주희_x좌표와 y좌표 값 배열 만들기 _200819
# for i in range(len(rect_center)):
#     if i%2 == 0:
#         center_x.append(rect_center[i])
#     if i%2 == 1:
#         center_y.append(rect_center[i])


# print(center_y[0:3])

# for i in range(len(center_x)):
#     if

# by 배아랑이_중심점 좌표 [x,y]로 표현하기 _200708
rect_center = np.array(rect_center)
# print(rect_center[0])
rect_center = np.reshape(rect_center, [int(rect_center.shape[0] / 2), 2])
# print(rect_center)

# by 배아랑이_중심점 오름차순으로 정렬하기 _200708
img_line = closing.copy()
sort_center = np.sort(rect_center, axis=0)
# print(sort_center)

# by 배아랑이_중심점 선 그리기 _200708
for i in range(sort_center.shape[0]):
    cv.line(img_line, (sort_center[i,0],0), (sort_center[i,0],img_line.shape[0]), (0,0,255))
    cv.line(img_line, (0, sort_center[i,1]), (img_line.shape[1],sort_center[i,1]), (0,0,255))
#
# plt.subplot(131)
# plt.imshow(img_thr, cmap='gray')
# plt.title('image by threshold')
# plt.axis('off')

# plt.subplot(132)
plt.imshow(closing, cmap='gray')
# plt.xlabel('장애인전')
plt.title('closing')
plt.axis('off')

print()
print('result : 장애인 전용 ')
# plt.subplot(133)
# plt.imshow(img_line, cmap='gray')
# plt.title('image with line')
# plt.axis('off')

plt.show()

