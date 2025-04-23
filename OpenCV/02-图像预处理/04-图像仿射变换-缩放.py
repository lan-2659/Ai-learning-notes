import cv2
import numpy as np
image = cv2.imread('./images/ceshi01.jpeg')
#获取图片的像素
(h,w) = image.shape[:2]
#定义缩放的大小，大于1的值就是放大，小于的值就是缩小
tx=0.5
ty=0.5
#构建一个平移矩阵
m = np.float32([[tx,0,0],[0,ty,0]])
#进行仿射变换
dst = cv2.warpAffine(image, m, (w*int(tx),h*int(ty)))
cv2.imshow('image01', image)
cv2.imshow('image02', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

