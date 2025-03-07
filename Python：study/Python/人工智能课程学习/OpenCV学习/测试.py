import cv2
import numpy as np

image = cv2.imread("images/1.jpg")

# image = np.fromfile(r'人工智能课程学习\OpenCV学习\img2.png', dtype=np.uint8)
# image = cv2.imdecode(image, cv2.IMREAD_COLOR)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()