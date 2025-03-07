import cv2

# 读取图片
image = cv2.imread('../../images/car3.png')

# 保存图片，会返回一个bool值，成功是True，失败是False
iss = cv2.imwrite("../save_imag/car2.png", image)
if iss:
    print("保存成功")
else:
    print("保存失败")