# 读取图片方法
"""
# 有两种方法用于读取图片
    1、使用cv2.imread()方法直接读取
    2、使用numpy.fromfile()方法读取，cv2.imdecode方法转换

    
# 有两种读取方法的原因：
    cv2.imread()方法 仅支持英文路径
    (对于路径中包含中文的图片会报错，无法读取)

    而numpy.fromfile()方法可以读取任意路径的文件为ndarray对象
    (但是这个对象不符合OpenCv中的图片格式所以需要转换)


# 第一种方法(也是最常用的)：
    cv2.imread()

        语法格式：
        cv2.imread(filename, flags)

            filename：读取的图片路径
            flags：读取图片的标志，默认为 cv2.IMREAD_COLOR

        注意事项：
        返回一个三维数组: (h, w, c)
        flags(常用的三个)：
            cv2.IMREAD_COLOR: 默认，读取为彩色图片，输出为三通道数组
            cv2.IMREAD_GRAYSCALE: 读取为灰度图片，输出为单通道数组
            cv2.IMREAD_ANYCOLOR: 根据图像自动选择彩色或灰度

        举例：
        import cv2

        image = cv2.imread("../images/car3.png")    # 将图片读取为 ndarray对象 

    
# 第二种方法(稍微麻烦，但适用于任意路径) 
    numpy.fromfile() + cv2.imdecode()  

        语法格式：
        image_bytes = numpy.fromfile(path, dtype=np.uint8)

            path：可以传入任意路径
            dtype：必须指定为 np.uint8，默认为 float
                     
        image = cv2.imdecode(image_bytes, flags)

            image_bytes：numpy.fromfile方法的返回值
            flags：读取图片的标志，默认为 cv2.IMREAD_COLOR

        注意事项：
        最终得到的 image 为符合OpenCv图像格式的 ndarray对象
        numpy.fromfile 方法的返回值是一维的NumPy数组，包含文件的原始二进制字节
        cv2.imdecode 方法将一维字节流转换为多维图像矩阵

        # 举例：
        import cv2 
        import numpy as np  

        image = np.fromfile('images/1.jpg', dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
"""


# 创建窗口方法
"""
cv2.namedWindow()

    语法格式：
    cv2.namedWindow(winname, flags)

        winname：创建的窗口的名字(一般为英文，用中文会有乱码)
        flags：窗口属性标志，默认为 cv2.WINDOW_AUTOSIZE

    注意事项：
    这个方法会创建一个窗口(这个窗口显示在桌面上，但是什么东西都没有，需要手动传入图片)
    flags(常用两个)：
        cv2.WINDOW_AUTOSIZE: 窗口会根据窗口内容自动调整大小，窗口尺寸无法手动更改
        cv2.WINDOW_NORMAL: 
            会根据窗口内容自动窗口调整大小，
            允许用户改变窗口大小(使用cv2.resizeWindow()方法，或拖动窗口边缘自由调整窗口大小),
            窗口内容会根据窗口尺寸自动缩放
"""


# 调整窗口尺寸方法
"""
cv2.resizeWindow()

    语法格式：
    cv2.resizeWindow(winname, width, height)

        winname：需要调整的窗口名称（必须已通过namedWindow创建）
        width：目标窗口宽度（像素单位）
        height：目标窗口高度（像素单位）

    注意事项：
    只有在创建窗口时使用cv2.WINDOW_NORMAL标志才有效
    调整的是窗口客户区尺寸（不包含标题栏和边框）
"""


# 显示图像方法
"""
cv2.imshow()

    语法格式：
    cv2.imshow(winname, mat)

        winname：显示图像的窗口名称
        mat：要显示的图像数组（numpy.ndarray类型）

    注意事项：
    若窗口不存在会自动创建（但此时窗口属性为默认的WINDOW_AUTOSIZE）
    实际显示时间取决于后续的cv2.waitKey()调用
"""


# 等待按键方法
"""
cv2.waitKey()

    语法格式：
    cv2.waitKey([delay])

        delay：等待时间（毫秒单位），默认0表示无限等待（输入为负数时与0等效）

    返回值：
    返回按键的ASCII码值（无按键时返回-1）

    注意事项：
    该方法用于捕获用户在激活窗口中的按键操作，刷新窗口，处理GUI事件(响应窗口移动、缩放、关闭等操作)
    无需须为每个窗口调用cv2.waitKey，但需要至少调用一次cv2.waitKey（如果不调用，很可能会导致窗口无响应或卡死）
"""


# 销毁窗口方法
"""
cv2.destroyAllWindows()

    语法格式：
    cv2.destroyAllWindows()

    功能说明：
    关闭所有通过OpenCV创建的窗口并释放相关资源

    注意事项：
    通常作为程序结束前的清理操作
    也可以使用cv2.destroyWindow(winname)销毁指定窗口
"""

# 使用OpenCv读取显示图片的全过程
import cv2

image = cv2.imread("../sre/1.jpg", cv2.IMREAD_COLOR)    # 将图片读取为numpy数组

cv2.namedWindow('images', cv2.WINDOW_NORMAL)            # 创建窗口，可以用于显示图片

cv2.resizeWindow('images', 1200, 800)                   # 调整窗口大小

cv2.imshow('images', image)                             # 显示图片

key = cv2.waitKey(0)                                    # 保持窗口显示，等待用户按键
if key == 13:
    cv2.destroyAllWindows()

