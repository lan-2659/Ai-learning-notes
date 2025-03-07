import cv2

# 开启摄像头
cap = cv2.VideoCapture("../../video/2.mp4")

# 判断收复开启开启摄像
isOpen = cap.isOpened()
if not isOpen:
    print("没有开启摄像头")

# 帧高
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("帧高:", h)
# 帧宽
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
print("帧宽:", w)
# 帧率
fpts = cap.get(cv2.CAP_PROP_FPS)
print("帧率", fpts)

# 计算视频播放频率
deplay = int(1000 / fpts)
while isOpen:

    # 这个方法会返回每一帧的图片
    # ret 是否读取到了数据帧
    # fram 数据帧或者图片
    ret, frame = cap.read()

    if ret:
        # 如果要改变视频的宽高度，采用窗口
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("frame", 500, 200)
        cv2.imshow('frame', frame)
    if cv2.waitKey(deplay) == 13:
        break

# 释放图片资源
cap.release()
cv2.destroyAllWindows()
