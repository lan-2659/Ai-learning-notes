import cv2
import numpy as np

image = cv2.imread('../images/car.png')

# 把RGB转换成HSV
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# 定义一个HSV颜色空间
lower = np.array([100, 100, 100])
height = np.array([140, 255, 255])

# 掩模处理
mask = cv2.inRange(image, lower, height)
cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
