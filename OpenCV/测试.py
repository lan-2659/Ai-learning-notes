import cv2 as cv
import numpy as np

# 读取图像
img = cv.imread("images/demo.png")

# 调整图像大小
img_np = cv.resize(img, (480, 480))

# 颜色空间转为HSV
hsv_img_np = cv.cvtColor(img_np, cv.COLOR_BGR2HSV)

# 修改蓝色范围
color_low = np.array([100, 46, 46])
color_high = np.array([124, 255, 255])

# 创建蓝色掩膜
mask_blue = cv.inRange(hsv_img_np, color_low, color_high)

# 颜色替换：将蓝色改为红色
img_np[mask_blue == 255] = (0, 0, 255)

# 显示结果（窗口名称保持原样）
cv.imshow("GR", img_np)
cv.imshow("hsv", hsv_img_np)
cv.imshow("mask", mask_blue)
cv.waitKey(0)
cv.destroyAllWindows()