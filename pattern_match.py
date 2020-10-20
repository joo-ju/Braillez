import cv2 as cv
import matplotlib.pyplot as plt


img = cv.imread('./data/k.jpg', cv.IMREAD_GRAYSCALE)

img_re = cv.resize(img, ( 2592, 1944))

plt.subplot(121)

plt.imshow(img, cmap='gray')
plt.title('Original image')
plt.axis('off')

plt.subplot(122)

plt.imshow(img_re, cmap='gray')
plt.title('resize image')
plt.axis('off')

plt.show()