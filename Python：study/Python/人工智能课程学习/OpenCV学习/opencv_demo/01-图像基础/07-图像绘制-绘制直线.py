import cv2

image = cv2.imread("../../images/car3.png")

# 起始坐标
start = (100, 100)
# 截止点坐标
end = (200, 200)
# 颜色
rgb = (255, 0, 0)
# 线条的宽度
t = 5

# 传入参数
cv2.line(image, start, end, rgb, t)

cv2.imshow("images", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
