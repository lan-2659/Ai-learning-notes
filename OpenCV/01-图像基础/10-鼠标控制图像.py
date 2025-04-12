import cv2


def draw_mouse(event, x, y, flags, param):
    """
    :param event: 鼠标当前做的事，比如：点击、移动等等
    :param x: 鼠标所在位置的x坐标
    :param y: 鼠标所在位置的y坐标
    :param flags: 不知道，用到了再回来补充
    :param param: 不知道，用到了再回来补充
    """
    # 鼠标按下事件
    if event == cv2.EVENT_LBUTTONDOWN:
        print("鼠标按下")
        cv2.circle(image, (x, y), 5, (255, 0, 0), 5)
        cv2.imshow("images", image)
    # 鼠标按下松开的事件
    elif event == cv2.EVENT_LBUTTONUP:
        print("鼠标按下松开")
    # 鼠标移动事件
    elif event == cv2.EVENT_MOUSEMOVE:
        print("鼠标移动了")


image = cv2.imread("../../images/car2.png")
# 创建窗口
cv2.namedWindow("images")

# 设置鼠标的回调函数
# 注意cv2.setMouseCallback()函数是基于窗口的，调用时必须传入一个创建好的窗口
# draw_mouse是一个函数名，cv2.setMouseCallback()会向draw_mouse中传入五个参数
# 所以draw_mouse函数的定义中至少包含五个形参
cv2.setMouseCallback("images", draw_mouse)

while True:
    cv2.imshow("images", image)
    if cv2.waitKey(1) == 13:
        # 保存图片
        iss = cv2.imwrite("../save_image/car2_new.png", image)
        if iss:
            print("图片保存成功")
        else:
            print("图片保存失败")
        break

cv2.destroyAllWindows()
