import cv2

image = cv2.imread("../../images/car3.png")
print("原图片的像素", image.shape)

# 调整图片大小。cv2.resize()函数用于图片；cv2.resizeWindow()用于窗口
img = cv2.resize(image, (300, 500))
print("改变后的图片的像素", img.shape)

cv2.imshow("image01", image)
cv2.imshow("image02", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

