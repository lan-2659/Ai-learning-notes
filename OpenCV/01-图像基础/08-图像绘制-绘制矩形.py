import cv2

image = cv2.imread("../../images/car3.png")

# 左上角的坐标
start = (0, 0)
# 右下角坐标
end = (200, 200)
# 颜色
rgb = (255, 0, 0)
# 线条的宽度，传入负数时会将整个矩形填满
t = 25

# 传入参数
cv2.rectangle(image, start, end, rgb, t)

cv2.imshow("images", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
