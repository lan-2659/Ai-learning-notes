import cv2
import numpy as np

image = cv2.imread('../../images/car6.png')

# 把BGR转换成HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# 定义一个HSV颜色空间
lower = np.array([100, 120, 100])
height = np.array([140, 255, 255])

# 掩模处理，只能传HSV图片
# 返回一个二值图像，白色部分表示在指定颜色范围内的区域，黑色部分表示不在范围内的区域
mask = cv2.inRange(hsv_image, lower, height)
cv2.imshow('mask', mask)

# 查找轮廓，只能传入二值图像，contours中存放所有的轮廓，hierarchy包含轮廓的层级信息，包含轮廓之间的关系
contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))

# 把灰度图像转换彩色图像
output_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
# 绘制轮廓，-1那个位置是索引号，传入负数表示绘制全部
cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)
cv2.imshow('output_image', output_image)

# cv2.boundingRect()函数 用于计算轮廓的最小外接矩形
# 返回一个元组(x, y, w, h)，(x, y)是矩形左上角的坐标，w是宽度，h是高度
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 50 and h > 20:
        print(x, y, w, h)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # 剪切：
        qie_image = image[y:y + h, x:x + w]

cv2.imshow('Contours', qie_image)
cv2.imwrite('../../img2.png', qie_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
