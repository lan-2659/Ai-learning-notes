import cv2

# 设置一个全局变量，记录是否开始画画
draw = False
# 记录移动的坐标
points = []
start = 0
endp = 0


def draw_mouse(event, x, y, flags, param):
    global draw, points
    # 鼠标按下事件
    if event == cv2.EVENT_LBUTTONDOWN:
        draw = True
    # 鼠标按下松开的事件
    elif event == cv2.EVENT_LBUTTONUP:
        draw = False
    # 鼠标移动事件
    elif event == cv2.EVENT_MOUSEMOVE:
        if draw:
            # 记录所有坐标
            points.append((x, y))
            for i in range(len(points)):
                print(points[i])
                # 起始坐标
                start = points[i]
                # 截止点坐标
                if i + 1 < len(points):
                    endp = points[i + 1]
                cv2.line(image, start, endp, (255, 0, 0), 5)


image = cv2.imread("../../images/car2.png")
# 创建窗口
cv2.namedWindow("images")

# 设置鼠标的回调函数
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
