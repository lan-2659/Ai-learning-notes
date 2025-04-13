import cv2

# 读取图像
image = cv2.imread('../../images/car4.png')
# 将图像从BGR到灰度图像的转换
gay_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 二值化处理，retval中存放真正使用的阈值，binary_image中存放二值化后图片的ndarray数组信息
"""
四个参数：
    src：
        二值化只能传入灰度图片
    thresh:
        阈值，决定分割的界限
    maxval:
        当像素值超过阈值时，赋予的最大值（通常为255）
    type:(阈值类型（常用的）)
        - cv2.THRESH_BINARY: 超过阈值的像素设为最大值，其余设为0。
        - cv2.THRESH_BINARY_INV: 超过阈值的像素设为0，其余设为最大值。
        - cv2.THRESH_TRUNC: 超过阈值的像素设为阈值，其余不变。
        - cv2.THRESH_TOZERO: 超过阈值的像素不变，其余设为0。
        - cv2.THRESH_TOZERO_INV: 超过阈值的像素设为0，其余不变。
"""
retval, binary_image = cv2.threshold(gay_image, 100, 255, cv2.THRESH_BINARY)
# cv2.imshow('binary', binary_image)

# 查找轮廓，contours中存放所有的轮廓，hierarchy包含轮廓的层级信息，包含轮廓之间的关系
"""
三个参数：
    image：
        只能传入二值图像
    mode：(轮廓检索方式)
        - cv2.RETR_EXTERNAL: 只检索外部轮廓。
        - cv2.RETR_LIST: 检索所有轮廓，并将其放入列表中。
        - cv2.RETR_TREE: 检索所有轮廓，并建立层级关系。
    method：(轮廓接近方式)
        - cv2.CHAIN_APPROX_SIMPLE: 压缩轮廓，仅保留端点。
        - cv2.CHAIN_APPROX_NONE: 保留所有轮廓点。
"""
contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# print(hierarchy)
# print(contours[0].shape)


# 把灰度图像转换彩色图像
output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
# cv2.imshow('output_image', output_image)

# 绘制轮廓，-1那个位置是索引号，传入负数表示绘制全部
cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)
cv2.imshow('image', output_image)

# cv2.boundingRect()函数 用于计算轮廓的最小外接矩形
# 返回一个元组(x, y, w, h)，(x, y)是矩形左上角的坐标，w是宽度，h是高度
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w == 203 and h == 56:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        image2 = image[y:y + h, x:x + w]

cv2.imshow('Contours', image)
cv2.imwrite('../../img2.png', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
