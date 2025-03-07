import cv2

image = cv2.imread("../images/car.png")
#转换成灰度图像
gay_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#看下效果
cv2.imshow("image", gay_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
