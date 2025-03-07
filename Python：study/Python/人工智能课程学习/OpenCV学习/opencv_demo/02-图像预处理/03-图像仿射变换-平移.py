import cv2
import numpy as np
image = cv2.imread('../images/fangshe.jpeg')
#获取图片的像素
(h,w) = image.shape[:2]
#定义平移参数，向左移动100px,向下移动50px
tx = 100
ty = 50
#构建一个平移矩阵
m = np.float32([[1,0,tx],[0,1,ty]])
print(m)
#进行仿射变换
dst = cv2.warpAffine(image, m, (w,h))
cv2.imshow('image01', image)
cv2.imshow('image02', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
