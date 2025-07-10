import cv2

image = cv2.imread("images/flower.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

cv2.imshow("image", image)
cv2.imshow("gray", gray)
cv2.imshow("binary", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()