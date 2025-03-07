import cv2

# 开启摄像头，传入0作为参数
cap = cv2.VideoCapture(0)

# 判断是否成功开启摄像头
isOpen = cap.isOpened()
if not isOpen:
    print("没有开启摄像头")

while isOpen:

    # 这个方法会返回每一帧的图片
    # ret 是否读取到了数据帧
    # fram 数据帧或者图片
    ret, frame = cap.read()

    if ret:
        cv2.imshow('frame', frame)
    if cv2.waitKey(1) ==13:
        break

# 释放图片资源
cap.release()
cv2.destroyAllWindows()
