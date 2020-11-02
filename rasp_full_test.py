import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
from collections import Counter
from braille import table_kor
from unicode import join_jamos
# table_kor = {'ㄱ': '000100', 'ㄴ':'100100', 'ㄷ':'010100', 'ㄹ':'000010', 'ㅁ':'1000010', 'ㅂ':'000110',
#              'ㅅ':'000001', 'ㅈ':'000101', 'ㅊ':'000011', 'ㅋ':'110100', 'ㅌ':'110010', 'ㅍ':'100110',
#              'ㅎ':'010110', '된소리':'000001', '_ㄱ':'100000', '_ㄴ':'010010', '_ㄷ':'001010', '_ㄹ':'010000',
#              '_ㅁ':'010001', '_ㅂ':'110000', '_ㅅ':'001000', '_ㅇ':'011011', '_ㅈ':'101000', '_ㅊ':'011000',
#              '_ㅋ':'011010', '_ㅌ':'011001', '_ㅍ':'010011', '_ㅎ':'001011', 'ㅆ':'001100',
#              'ㅏ':'110001', 'ㅑ':'001110', 'ㅓ':'011100', 'ㅕ':'100011', 'ㅗ':'101001', 'ㅛ':'001101','ㅜ':'101100',
#              'ㅠ':'101100', 'ㅡ':'010101', 'ㅣ':'101010', '_ㅣ':'111010', 'ㅐ':'111010', 'ㅔ':'101110',
#              'ㅖ':'001100', 'ㅘ':'111001', 'ㅚ':'101111', 'ㅝ':'111100', 'ㅢ':'010111',
#              '가':'110101', '사':'111000', '억':'100111', '언':'011111', '얼':'011110', '연':'100001',
#              '열':'110011', '영':'110111', '옥':'101101', '온':'111011', '옹':'111111', '운':'110110',
#              '울':'111101', '은':'101011', '을':'011101', '인':'111110'}

# 카메라로 사진을 다시 찍어서 실행해봄. 명암이 균일하지 않아서 검출에 힘들었던 사진말고 조금더 명암이 균일한 조건에서 촬영함
# 조건 : 표지판과 카메라 10cm 거리, 카메라와 표지판이 평행하게 촬영


# img = cv.imread('./data/example.png', cv.IMREAD_GRAYSCALE)
# img = cv.imread('./data/example_1.jpeg', cv.IMREAD_GRAYSCALE)
# img = cv.imread('./data/smoking.png', cv.IMREAD_GRAYSCALE)
# img = cv.imread('./data/smoking_2.jpg', cv.IMREAD_GRAYSCALE)
# img = cv.imread('./data/smoking_3.png', cv.IMREAD_GRAYSCALE)
# img = cv.imread('./data/g.jpg', cv.IMREAD_GRAYSCALE)
# img = cv.imread('./data/f.jpg', cv.IMREAD_GRAYSCALE)
# img = cv.imread('./data/e.jpg', cv.IMREAD_GRAYSCALE)
# img = cv.imread('./data/i.jpg', cv.IMREAD_GRAYSCALE)
# img = cv.imread('./data/k.jpg', cv.IMREAD_GRAYSCALE)
# img = cv.imread('./data/m.jpg', cv.IMREAD_GRAYSCALE)
# img = cv.imread('./data/n.jpg',cv.IMREAD_GRAYSCALE)
# img = cv.imread('./data/n.jpg')
# img = cv.imread('./data/o.jpg')
img = cv.imread('./data/aa.jpg')

# img_2 = cv.imread('./data/smoking.png')
# img_2 = cv.imread('./data/smoking_2.jpg')
# img_2 = cv.imread('./data/smoking_3.png')
# img_2 = cv.imread('./data/g.jpg')
# img_2 = cv.imread('./data/f.jpg')
# img_2 = cv.imread('./data/e.jpg')
# img_2 = cv.imread('./data/k.jpg')
# img_2 = cv.imread('./data/m.jpg')
# img_2 = cv.imread('./data/o.jpg')
# img_2 = cv.imread('./data/n.jpg')
# img_2 = cv.imread('./data/j.jpg')
img_2 = cv.imread('./data/aa.jpg')


# by 김주희_이미지 사이즈 변경_201020
img = cv.resize(img, (2592, 1944))
img_2 = cv.resize(img_2, (2592, 1944))

# 커널 생성(대상이 있는 픽셀을 강조)
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

# 커널 적용
img_A = cv.filter2D(img, -1, kernel)
# image_enhanced = cv.equalizeHist(img)
test_img = img.copy()
test_img2 = img_A.copy()

# 컨투어 영역의 넓이 중 최빈값으 구하기 위한 배
compare_area = []

yuv = cv.cvtColor(img_A, cv.COLOR_BGR2YUV)

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
closing[:cut_num[0]*2, :] = 125
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
    if (aspect_ratio > 0.8) and (aspect_ratio < 1.2):
        # 크기가 10픽셀 이하인 것은 모두 없애기._201012
        if (w > 10) and (w < 45):
            cv.rectangle(closing, (x, y), (x + w, y + h), (0, 255, 0), 2)
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

dx2 = center[0][0] - center[5][0]
dy2 = center[0][1] - center[5][1]

print('dx', dx)
print('dy', dy)
print('dx2', dx2)
print('dy2', dy2)

# by 김주희_두 기울기의 평균으로 회전하기_201019
ave = math.atan(dy2/dx2)
# ave = (math.atan(dy/dx) + math.atan(dy2/dx2)) / 2
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
    if (aspect_ratio > 0.8) and (aspect_ratio < 1.2):
        # 크기가 10픽셀 이하인 것은 모두 없애기._201012
        if (w > 5) and (w < 25):
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
center_y.sort()
print("정렬된 center_x : ", center_x)
print("정렬된 center_y : ", center_y)
diff = 0
count = 0
sum = 0
count_x = 0
sum_x = 0
hline = []
vline = []

sameLevel_y = []
sameLevel_x = []
temp = []
temp2 = []
mid_x = []
mid_y = []
index = 0   # sameLevel 변수의 레벨을 나누는 것. (같은 라인인지 아닌지 구분)

# by 김주희_ 거리를 구할 때 인덱스오류가 나서 하나 더 대입해줌 - 영향 없음_201019
center_y.append(0)
center_x.append(0)
# center_y.insert(0, 0)
# center_x.insert(0, 0)

print("추가한 center_x : ", center_x)
print("추가한 center_y : ", center_y)

# by 김주희_y값 기준 만약에 center의 거리가 20px이하이면 카운트 해주고, 값을 더해준다. (나중에 평균 구할 예정) _201019
for i in range(len(center_y)-1):
    # by 김주희_x값중 쵀대값과 최소값의 중간값 구하기-더해서 나누기2 _201020
    # 중심값의 x값의 차이가 20 이하이면 같은 배열에 대입한다.

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



# plt.subplot(211)
# plt.subplot(121)
# plt.imshow(img_2, cmap='gray')
# plt.title('Original Image')
# plt.axis('off')


print("mid_x : ", mid_x)
print("mid_y : ", mid_y)

# by 김주희_수평선 그리기_201021
for i in range(len(mid_y)):
    plt.hlines(mid_y[i]-20, 0, 2592, colors='lightblue', linewidth=1)
    hline.append(mid_y[i]-20)
    # plt.hlines(mid_y[i]+20, 0, 2592, colors='lightblue', linewidth=1)
    if i == len(mid_y)-1:
        plt.hlines(mid_y[i]+20, 0, 2592, colors='lightblue', linewidth=1)
        hline.append(mid_y[i]+20)

# by 배아랑이_점자 영역 구분하는 수직선, 수평선_201023
hline.sort()
print("hline : ", hline)


braille = []
count = 0
mid_x.append(0)
mid_x.append(0)
# by 배아랑이_점자 영역 구분_201023
for i in range(len(mid_x)-2):
    if 75 <= (mid_x[i + 1] - mid_x[i]):
        if 120 <= (mid_x[i + 1] - mid_x[i]):
            if 150 <= (mid_x[i + 1] - mid_x[i]):
                # 1___1 -> 점자 2개
                if i != len(mid_x)-3:
                    braille.append([mid_x[i], mid_x[i]+40])
            else:
                # 1__1 -> 점자 2개  -> 2가지 1 _ _1 / 1_ _ 1
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
        # 11 -> 점자 하나
        if i == len(mid_x)-3:
            braille.append([mid_x[i-1], mid_x[i]])


print("braille", braille)

# mid_x = []
# mid_y = []

# by 김주희_6등분 선중 수직선 좌표과 저장되어 있응 braille을 이용하여 수직선 그리기 & vline 리스트에 선 좌표 넣_201024
vvline=[]
for i in range(len(braille)):
    plt.vlines(braille[i][0]-20, 0, 1944, colors='lightblue', linewidth=1)
    plt.vlines(braille[i][1]-20, 0, 1944, colors='lightblue', linewidth=1)
    plt.vlines(braille[i][1]+20, 0, 1944, colors='lightblue', linewidth=1)
    vline.append(braille[i][0]-20)
    vline.append(braille[i][1]-20)
    vline.append(braille[i][1]+20)


print("vline : ", vline)

vline.append(max(vline)+40)
vline.append(max(vline)+40)
vline.append(max(vline)+40)
print("vline : ", vline)
ro_center.sort()
print("정렬된 ro_center : ", ro_center)
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

    print(j)

binary_code.append(b)
print("binary_code", binary_code)
# 숫자형 리스트를 문자열로 바꿔 저장하는 변
kor_d = []
#by 김주희_점자의 점 위치를 파악한 배열을 이용하여 바이너라 형식으로 변환_201024
kor_b = np.zeros((len(binary_code), 6),np.int32)
print("kor_b " ,kor_b)
for i in range(len(binary_code)):
    for j in range(len(binary_code[i])):
        kor_b[i][binary_code[i][j]] = 1
    a = ''.join(map(str, kor_b[i]))
    print(" a : " , a)
    kor_d.append(a)
    # kor[i] = ''. join(map(str, kor[i]))

    # print(str(int('', join(kor[i]))))

print("str kor " ,kor_b)
print("kor_d", kor_d)

table_kor_v = {v: k for k, v in table_kor.items()}
print("table_kor_v : ", table_kor_v)
# by 김주희__201101


# by 김주희_dictionary 형식에 인덱싱하기_201101
for index, (key, value) in enumerate(table_kor_v.items()):
  print(index, key, value)

print("Total Price : {} ".format(list(table_kor_v.values())[0]))

print("table_kor_v 인덱싱 이후 : ", table_kor_v)
print(index)
# by 김주희_table_kor_v에서 매칭시킨 한글을 순서대로 저장하는 리스트_201101
string =[]

# by 김주희_한글자씩 정확하게 만들어서 저장하는 리스트 _201101
char=[]

# by 김주희_인덱스로 접근하기 위해 리스트 형식으로 만든다._201101
list_char = list(table_kor_v.values())
print("list_char : ", list_char)

# by 김주희_각 글자의 인덱스만 저장하는 리스트 _201101
index_char = []

# while i <= len(kor_d):
# # for i in range(len(kor_d)):
#     s = table_kor_v.get(kor_d[i])
#     print(table_kor_v.get(kor_d[i]))
#     string.append(s)
#     index_char.append(list_char.index(s))
#
#     # by 김주희_받침인 경우 _없애기 _201101
#
#     # by 김주희_한글자씩 나누기 _201101
#     if i!=0:   # 처음이 아닐 때
#         # by 김주희_ 자음_받침 일 경우 -> 자음 ㅏ 받침 _201101
#         if (0 <= index_char[i-1] <= 12) and (13 <= index_char[i] <= 26):
#             string.insert(i,'ㅏ')
#             i = i + 1
#             continue
#
#         # by 김주희_ 받침_모음 일 경우 -> 받침 ㅇ 모음 _201101
#         elif (13 <= index_char[i-1] <= 26) and (27 <= index_char[i] <= 42):
#             string.insert(i,'ㅇ')

j = 0
# for i in range(len(kor_d)):
#
#     s = table_kor_v.get(kor_d[i])
#     print(table_kor_v.get(kor_d[i]))
#     string.append(s)
#     index_char.append(list_char.index(s))

    # by 김주희_받침인 경우 _없애기 _201101

    # # by 김주희_한글자씩 나누기 _201101
    # if j < len(string):   # 처음이 아닐 때
    #     # by 김주희_ 자음_받침 일 경우 -> 자음 ㅏ 받침 _201101
    #     if (0 <= index_char[j] <= 12) and (13 <= index_char[j+1] <= 26):
    #         string.insert(j,'ㅏ')
    #         j = j + 2
    #         print("string = ", string)
    #
    #
    #
    #     # by 김주희_ 받침_모음 일 경우 -> 받침 ㅇ 모음 _201101
    #     if (13 <= index_char[j] <= 26) and (27 <= index_char[j+1] <= 42):
    #         string.insert(j,'ㅇ')
    #         j = j + 2
    #         print("string = ", string)

# by 김주희_점자 자음과 모음 맞춰서 배열 만들기_201102
# index_char와 string의 인덱스가 다르므로 구분해야함.
# string에는 새로운 문자가 들어가지만 lndex_char에는 들어가지 않음.
# index_char에도 string처럼 값을 넣어줘야 한다.

# j 는 index_char와 string의 인덱스로 쓰이고
# m 은 for문 반복 횟수
j = 0
for m in range(len(string)-1):
    # by 김주희_한글자씩 나누기 _201101
    # by 김주희_ 자음_받침 일 경우 -> 자음 ㅏ 받침 _201101
    if (0 <= index_char[j] <= 12) and (13 <= index_char[j+1] <= 26):
        string.insert(j+1,'ㅏ')
        index_char.insert(j+1, list_char.index('ㅏ'))
        # index_char.insert(j+1, table_kor.)
        print("j 1 = ", j)
        j = j + 2
        print("string 1 = ", string)
        print("index_char : ", index_char)


    # by 김주희_ 받침_모음 일 경우 -> 받침 ㅇ 모음 _201101
    elif (13 <= index_char[j] <= 26) and (27 <= index_char[j+1] <= 42):
        print("j 2 = ", j)
        string.insert(j+1,'ㅇ')
        index_char.insert(j+1, -1)
        # index_char.insert(j+1, list_char.index('ㅇ'))
        j = j + 2
        print("string 2 = ", string)

    # by 김주희_ 받침/모음_줄임말 일 경우 -> 받침/모음 ㅇ 줄임말 _201102
    elif ((27 <= index_char[j] <= 42) and (45 <= index_char[j+1] <= 58)) or ((13 <= index_char[j] <= 26) and (45 <= index_char[j+1] <= 58)):
        print("j 3 = ", j)
        string.insert(j+1,'ㅇ')
        index_char.insert(j+1, -1)
        # index_char.insert(j+1, list_char.index('ㅇ'))
        j = j + 2
        print("j 3-1 = ", j)
        print("string 3 = ", string)

    # by 김주희_ 받침/모음/줄임말_모음 일 경우 -> 받침/모음/줄임말 ㅇ 모음 _201102
    elif ((27 <= index_char[j] <= 42) and (27 <= index_char[j+1] <= 42)) or ((13 <= index_char[j] <= 26) and (27 <= index_char[j+1] <= 42)):
        print("j 4 = ", j)
        string.insert(j + 1, 'ㅇ')
        index_char.insert(j + 1, -1)
        j = j + 2
        print("string 4 = ", string)
    elif ((45 <= index_char[j] <= 58) and (27 <= index_char[j+1] <= 42)):
        print("j 4 = ", j)
        string.insert(j + 1, 'ㅇ')
        index_char.insert(j + 1, -1)
        j = j + 2
        print("string 4 = ", string)
    else:
        j = j+1
        print("nothing j = ", j)


print("string = ", string)
print("index_char = ", index_char)


str_full = ''.join(string)
print(str_full)





print(join_jamos(str_full))

if kor_d[0] == '000101':
    str = "장애인전용"
if kor_d[0] == '100011':
    str = "여자화장실"


# for i in range(len(vline)):
#     for j in range(len(hline)):
#
# if rotation[vline[0]:vline[1], hline[2]:hline[3]].any():
#     print('braille')
# print(rotation[386, 688])
# print(rotation[402, 827])
# print(rotation[242, 738])

# print("vline size : ", len(vline))
# for i in range(len(vline)-1):
#     for j in range(len(hline)-1):
#         if (rotation[vline[i]+5:vline[i+1]-5, hline[j]+5:hline[j+1]-5]).any([255,255,255]):
#             braille.append(1)
#         else:
#             braille.append(0)
#     braille.append([i])
# print(braille)

# print(vline[11], vline[12], vline[13])
#
# if (rotation[vline[11]:vline[12], hline[1]:hline[2]]==[255,255,255]).any():
#     print('1')
# else:
#     print('0')


# #
# plt.subplot(222)
# plt.subplot(122)
# # plt.imshow(equ, cmap='gray')
# # plt.title('histogram equalization')
# plt.imshow(closing)
# plt.title('closing')
# plt.axis('off')

# plt.hlines(center[0][1], 0, 2592, colors='pink', linewidth=1)

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
plt.subplot(221)
plt.imshow(rotation)
plt.title('rotation Image')
plt.axis('off')
plt.subplot(222)
plt.imshow(test_img)
plt.title('test_img Image')
plt.axis('off')
plt.subplot(223)
plt.imshow(y2)
plt.title('y2 Image')
plt.axis('off')
# plt.subplot(224)
# plt.imshow(image_enhanced)
# plt.title('image_enhanced Image')
# plt.axis('off')

plt.show()