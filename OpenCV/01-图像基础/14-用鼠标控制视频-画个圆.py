import cv2

draw = False
points = []


def draw_circle(event, x, y, flags, param):
    global draw
    if event == cv2.EVENT_LBUTTONDOWN:
        draw = True
        points.append((x, y))
    if event == cv2.EVENT_RBUTTONUP:
        draw = False


cap = cv2.VideoCapture("../../video/1.mp4")
if not cap.isOpened():
    print("没有读取到")

# 设置窗口
cv2.namedWindow("frame")
# 设置鼠标的回调函数
cv2.setMouseCallback("frame",
                     draw_circle)
while True:
    ret, frame = cap.read()
    if ret:
        if points:
            # 画圆
            for point in points:
                cv2.circle(frame, point, 20, (255, 0, 0), 2)
        cv2.imshow("frame", frame)
    if cv2.waitKey(1) == 13:
        break
cap.release()
cv2.destroyAllWindows()
