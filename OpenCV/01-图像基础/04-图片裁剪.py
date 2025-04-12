import cv2

image = cv2.imread("../../images/car3.png")

# 设置裁剪的参数
x, y, w, h = 0, 0, 200, 400

# 获取图片的宽度，高度
height, width, c = image.shape
print(height, width, c)

# 裁剪
if y+h < height and x+w < width:

    """img是一个ndarray数组，这个地方实际上是在进行数组的切片"""
    img = image[y:y+h, x:x+w]

    cv2.imshow("image01", image)
    cv2.imshow("image02", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("索引越界")