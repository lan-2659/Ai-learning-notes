import cv2
import numpy as np
image = cv2.imread('./images/renwu01.jpeg')
#获取图片的像素
(h,w) = image.shape[:2]
#定义剪切的的x轴和y轴比例，小于 1的值
tx = 0.2
ty = 0.2
#构建一个平移矩阵
m = np.float32([[1,tx,0],[ty,1,0]])
#进行仿射变换
dst = cv2.warpAffine(image, m)
cv2.imshow('image01', image)
cv2.imshow('image02', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
