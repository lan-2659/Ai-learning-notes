import cv2

# 读取图像
image = cv2.imread('../../images/shenfen03.jpg')

# 转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用 Canny 边缘检测，返回检测到的边缘图像，边缘部分为白色，其他部分为黑色。
# image：输入图像，必须是灰度图像。
# threshold1：第一个阈值，用于确定弱边缘。在这个例子中是 50。
# threshold2：第二个阈值，用于确定强边缘。在这个例子中是 80。
"""
像素的梯度值：
    非边缘  threshold1  弱边缘  threshold2  强边缘
                         ↓
          （与强边缘相连就判定为边缘，否则判定为非边缘）
"""
edges = cv2.Canny(gray_image, threshold1=50, threshold2=80)

# 显示原图和边缘检测结果
cv2.imshow('Original Image', image)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
