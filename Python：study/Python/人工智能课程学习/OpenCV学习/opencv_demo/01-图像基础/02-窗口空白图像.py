import cv2
import numpy as np

# 图像的像素
width, height, c = 640, 480, 3
image = np.zeros((width, height, c), np.uint8)
print(image)
cv2.imshow("images", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
