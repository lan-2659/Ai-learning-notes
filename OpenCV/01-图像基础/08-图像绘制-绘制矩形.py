
# cv2.rectangle()方法  
"""  
    语法格式：  
    cv2.rectangle(img, pt1, pt2, color, thickness=1, lineType=cv2.LINE_8, shift=0)  

        参数说明：  
        img: 输入图像（numpy数组，在该图像上绘制矩形）  
        pt1: 矩形的第一个顶点坐标（左上角顶点），格式为 (x, y)（整数坐标）  
        pt2: 矩形的第二个顶点坐标（右下角顶点），格式为 (x, y)（整数坐标）  
        color: 矩形颜色，BGR格式（例如 (255,0,0) 表示蓝色）  
        thickness: 矩形边框粗细，默认值为1，必须为整数  
            零或正数：绘制空心矩形，边框粗细为指定像素值  
            负数（如-1）：绘制实心矩形（填充整个矩形区域）  
        lineType: 线条类型（可选参数，默认值为cv2.LINE_8，支持cv2.LINE_4、cv2.LINE_AA抗锯齿）  
        shift: 对坐标进行右移操作（等价于坐标 >> shift，用于亚像素精度定位，默认0）  


    注意事项：  
    无返回值，直接在输入图像 `img` 上绘制矩形（原地修改）  


    举例：  
    import cv2  
    import numpy as np  

    # 创建512x512的黑色画布（BGR三通道）  
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)  

    # 绘制红色空心矩形（左上角(50,50)，右下角(450,450)，线宽5，默认LINE_8）  
    cv2.rectangle(canvas, (50, 50), (450, 450), (0, 0, 255), 5)  

    # 绘制绿色实心矩形（左上角(100,100)，右下角(400,400)，填充颜色）  
    cv2.rectangle(canvas, (100, 100), (400, 400), (0, 255, 0), -1)  

    # 绘制蓝色抗锯齿空心矩形（左上角(150,150)，右下角(350,350)，线宽2，LINE_AA）  
    cv2.rectangle(canvas, (150, 150), (350, 350), (255, 0, 0), 2, cv2.LINE_AA)  

    # 绘制带坐标偏移的矩形（实际坐标为(200/2, 200/2)=(100,100)到(400/2,400/2)=(200,200)，shift=1）  
    cv2.rectangle(canvas, (200, 200), (400, 400), (100, 200, 250), 3, shift=1)  

    cv2.imshow("Rectangle Demo", canvas)  
    cv2.waitKey(0)  
    cv2.destroyAllWindows()  
"""