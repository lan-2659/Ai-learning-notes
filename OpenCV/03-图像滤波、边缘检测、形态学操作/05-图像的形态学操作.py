import cv2
import numpy as np

def test01():
    """腐蚀"""
    # 读取图像
    img = cv2.imread("../../images/shenfen03.jpg")
    # 定义结构元素
    kernel = np.ones((3, 3), np.uint8)
    # 执行腐蚀操作
    eroded = cv2.erode(img, kernel=kernel, iterations=3)
    # 显示结果
    cv2.imshow('Original Image', img)
    cv2.imshow('Eroded Image', eroded)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test02():
    """膨胀"""
    # 读取图像
    img = cv2.imread("../../images/shenfen03.jpg")
    # 定义结构元素
    kernel = np.ones((5, 5), np.uint8)
    # 执行膨胀操作
    eroded = cv2.dilate(img, kernel, iterations=3)
    # 显示结果
    cv2.imshow('Original Image', img)
    cv2.imshow('Eroded Image', eroded)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# 开运算
# 闭运算
# 梯度
# 顶帽
# 黑帽
if __name__ == '__main__':
    # test01()    # 腐蚀
    test02()    # 膨胀
