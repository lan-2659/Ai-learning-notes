import cv2

image = cv2.imread("../../images/car3.png")

# 设置圆心坐标
x = (100, 100)
# 半径
radius = 100
# 颜色
rgb = (255, 0, 0)
# 线条的宽度，传入负数时则会将整个圆填满
t = 5

# 传入数据
cv2.circle(image, x, radius, rgb, t)

cv2.imshow("images", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
