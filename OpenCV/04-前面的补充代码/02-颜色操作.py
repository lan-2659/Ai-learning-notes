import cv2
import numpy as np

img = cv2.imread("./src/pig.png")
img2 = cv2.imread("./src/cao.png")

# 由于cv2.imread()方法读取图片返回的是一个ndarray数组，所以可以直接进行加法操作
print(img + img2)
cv2.imshow("statck", img + img2)

# 但是直接加可能会导致像素的取值范围超出0-255，所以需要使用cv2.add()方法
img3 = cv2.add(img, img2) # 如果结果超出范围会对结果进行取余（对256取余）
# cv2.addWeighted()方法可以设置权重
img4 = cv2.addWeighted(img, 0.5, img2, 0.9, 0)

cv2.imshow("img3", img3)
cv2.imshow("img4", img4)
cv2.imshow("img", img)
cv2.imshow("img2", img2)

cv2.waitKey(0)