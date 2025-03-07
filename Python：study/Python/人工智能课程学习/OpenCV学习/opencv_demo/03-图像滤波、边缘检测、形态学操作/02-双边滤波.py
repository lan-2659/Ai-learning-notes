""""""
"""
原理
双边滤波器同时考虑两个因素：空间距离和像素值的相似度。具体来说，它使用两个高斯函数：
    空间高斯权重：基于像素在空间上的距离，距离越近的像素权重越大。
    像素值高斯权重：基于像素值之间的差异，颜色差异越小的像素权重越大。
最终，每个像素的输出值是其邻域内所有像素值的加权平均，权重是上述两个高斯权重的乘积。

双边滤波主要用于图像去噪、边缘保持（相比于其它滤波方式效果更好）、平滑均匀区域
特别适合需要在去噪的同时保留边缘的场合
"""

import cv2

# 读取图像
image = cv2.imread('../../images/shenfen03.jpg')

# 参数：d（邻域直径）、sigmaColor（灰度值差异的标准差）、sigmaSpace（空间距离的标准差）
blurred_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

# 显示原图和滤波后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Blurred Image', blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
