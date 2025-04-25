import cv2
import numpy as np

image = cv2.imread('images/1.jpg')
# cv2.imshow('image', image)

旋转矩阵 = cv2.getRotationMatrix2D((0, 0), 45, 1)
旋转矩阵 = np.vstack((旋转矩阵, [0, 0, 1]))
原始矩阵 = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]], dtype=np.float32)
平移矩阵 = np.array([[1, 0, 100], [0, 1, 100], [0, 0, 1]], dtype=np.float32)

m = np.dot(平移矩阵, 原始矩阵)
# m = np.dot(m, 旋转矩阵)
image1 = cv2.warpAffine(image, m[:2], (image.shape[1], image.shape[0]))
cv2.imshow('image1', image1) 

M  = np.linalg.inv(m)
image2 = cv2.warpAffine(image1, M[:2], (image.shape[1], image.shape[0]))
cv2.imshow('image2', image2)


# m = np.dot(平移矩阵, 原始矩阵)
# m = np.dot(旋转矩阵, m)
# image2 = cv2.warpAffine(image, m[:2], (image.shape[1], image.shape[0]))
# cv2.imshow('image2', image2) 

cv2.waitKey(0)
cv2.destroyAllWindows()
