import cv2

image = cv2.imread('./images/renwu01.jpeg')
#获取图片的像素
(h,w) = image.shape[:2]
print((h,w) )
#计算旋转坐标，中心点坐标
center=(0, 0)
print(center)
#旋转的角度
du = 45
#获取旋转矩阵,1 表示不缩放，原始大小
m = cv2.getRotationMatrix2D(center,du,1)
print(m)
#进行仿射变换
image2 = cv2.warpAffine(image,m,(w,h))
cv2.imshow("image1",image)
cv2.imshow('image2',image2)
#保存旋转后的图片
r = cv2.imwrite("../images/fangshe.jpeg",image2)
if r:
    print("OK")
else:
    print("Failed")

cv2.waitKey(0)
cv2.destroyAllWindows()


