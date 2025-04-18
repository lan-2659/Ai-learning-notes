# 添加水印
import cv2
import numpy as np

# logo = cv2.imread("./src/logohq.png")
logo = np.fromfile(r'D:\Python：study\Python\人工智能课程学习\OpenCV学习\opencv_demo\04-前面的补充代码\src\logohq.png', dtype=np.uint8)
logo = cv2.imdecode(logo, cv2.IMREAD_COLOR)

# img = cv2.imread("./src/bg.png")
img = np.fromfile(r'D:\Python：study\Python\人工智能课程学习\OpenCV学习\opencv_demo\04-前面的补充代码\src\bg.png', dtype=np.uint8)
img = cv2.imdecode(img, cv2.IMREAD_COLOR)

# 得到白化的logo(logo变白 其他区域变黑) ==> 为了抠出红色的文字
logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
_, mask_whitelogo = cv2.threshold(logo_gray, 180, 255, cv2.THRESH_BINARY_INV)

"""
两种方法提取logo：
    1、把原图和自己进行按位与，把白化的logo作为掩膜 ==> 只留下了文字其他地方全是黑色的BGR图
    2、对原图进行处理，用 mask_whitelogo==255 这个条件在原图中将对应的区域全部换成白色
"""
logo2 = logo.copy()
logo2[mask_whitelogo != 255] = [0, 0, 0]
logo_roi = cv2.bitwise_and(logo, logo, mask=mask_whitelogo) # 方法一提取logo
# cv2.imshow('logo_roi', logo_roi)
logo_roi2 = cv2.bitwise_and(logo, logo2) # 方法二提取logo
# cv2.imshow('logo_roi2', logo_roi2)


# ROI出img背景图的矩形区域(区域大小跟logo一致)
h, w, BRG = logo.shape
img_roi = img[100 : 100 + h, 100 : 100 + w]
# 得到黑化的logo(logo变黑 其他区域变白)==>为了抠出img_roi中的非文字区域
_, mask_blacklogo = cv2.threshold(logo_gray, 180, 255, cv2.THRESH_BINARY)
img_roi_ = cv2.bitwise_and(img_roi, img_roi, mask=mask_blacklogo)

shuiyin = cv2.add(logo_roi, img_roi_)
img_roi[::] = shuiyin

# cv2.imshow("mask_whitelogo", mask_whitelogo)
# cv2.imshow("logo_gray", logo_gray)
# cv2.imshow("logo", logo)
# cv2.imshow("logo_roi", logo_roi)
# cv2.imshow("img_roi", img_roi)
# cv2.imshow("mask_blacklogo", mask_blacklogo)
# cv2.imshow("img_roi_", img_roi_)
# cv2.imshow("img", img)
cv2.waitKey(0)



