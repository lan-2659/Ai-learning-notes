import cv2

#读取图片
image = cv2.imread('../images/car.png')
#0 :垂直翻转  1:水平翻转
new_image = cv2.flip(image,1)

cv2.imshow('image01', image)
cv2.imshow('image02', new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()