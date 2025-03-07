import cv2

# 读取图片，返回一个三维数组: (height, width, channels)
# 其中channels通常是3(BGR)或4(BGRA)
image = cv2.imread("../../images/car2.png")
print(image.shape)

# 创建窗口，第一个参数是窗口名称，一般是英文，如果是中文则会乱码，第二个参数可选，用来设置窗口大小
cv2.namedWindow('images', cv2.WINDOW_NORMAL)

# 设置窗口大小，只有当传入了 cv2.WINDOW_NORMAL 参数，才会设置成功
cv2.resizeWindow('images', 1200, 800)

# 显示图片，显示时间很短
# 如果窗口没有提前建立好，这个函数会根据图片大小现场建立一个窗口，无法更改现场建立的窗口的大小
cv2.imshow('images', image)

# 延时，持续到接收到用户对图片的键盘输入或时间结束，单位：ms，如果是0或负数则无限延长时间
# 会返回一个整数，代表用户按下的按钮
key = cv2.waitKey(0)
print(key)

if key == 13:
    print("进入")
    # 关闭全部窗口或者释放资源
    cv2.destroyAllWindows()

