import cv2
 
image = cv2.imread('images/1.jpg')
cv2.imshow('image', image)
image = cv2.flip(image, 0)

cv2.imshow('later', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
