"""
形态学变换：
    腐蚀、膨胀
    开运算、闭运算
    顶帽、黑帽
    形态学梯度
"""
import cv2
import numpy as np

# 创建一个7x7的灰度图片矩阵
arr = np.arange(0, 49).reshape((7, 7)).astype(np.uint8)
# print(arr)
# print()

# 初始化一个3x3的核矩阵，用于在被核矩阵覆盖的元素上进行筛选
k = np.array([[0, 1, 0], 
              [100, 100, 100], 
              [0, 100, 0]], dtype=np.uint8)

image_dilate = cv2.dilate(arr, k)   # 膨胀
image_erode = cv2.erode(arr, k)     # 腐蚀
# print('膨胀')
# print(image_dilate)
# print('腐蚀')
# print(image_erode)

image = cv2.imread('../../images/kai.jpg')
# cv2.imshow('image', image)

kernel = np.ones((5, 5), np.uint8)  # 创建一个5x5的核矩阵
# image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)     # 开运算
# cv2.imshow('image', image)

# image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel, iterations=2)   # 顶帽
# cv2.imshow('image', image)

cv2.imshow('dilate', cv2.dilate(image.copy(), kernel))
cv2.imshow('erode', cv2.erode(image.copy(), kernel))

image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)     # 形态学梯度
cv2.imshow('image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()