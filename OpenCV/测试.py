import cv2 
import numpy as np  

image = np.fromfile('OpenCv/images/1.jpg', dtype=np.uint8)
image = cv2.imdecode(image, cv2.IMREAD_COLOR)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
