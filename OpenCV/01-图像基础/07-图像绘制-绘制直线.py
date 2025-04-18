# 直线绘制
"""

# cv2.line()方法  

    语法格式：  
    cv2.line(img, pt1, pt2, color, thickness=1, lineType=cv2.LINE_8, shift=0)  

        参数说明：  
        img: 输入图像（numpy数组，在该图像上绘制直线）  
        pt1: 直线起点坐标，格式为 (x, y)（整数坐标）  
        pt2: 直线终点坐标，格式为 (x, y)（整数坐标）  
        color: 直线颜色，BGR格式（例如 (255,0,0) 表示蓝色）  
        thickness: 线条粗细，默认值为1 ，必须传入正整数 (0也不行)
        lineType: 线条类型（可选参数，默认值为cv2.LINE_8，支持cv2.LINE_4、cv2.LINE_AA抗锯齿）  
        shift: 对坐标进行 右移操作（等价于 坐标>>shift，然后用新的坐标进行绘制） 

        
    注意事项：  
    无返回值，直接在输入图像 `img` 上绘制直线（原地修改）

    
    举例：  
    import cv2  
    import numpy as np  

    # 创建512x512的黑色画布（BGR三通道）  
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)  

    # 绘制红色直线（起点(50,50)，终点(450,50)，线宽3，默认LINE_8）  
    cv2.line(canvas, (50, 50), (450, 50), (0, 0, 255), 3)  

    # 绘制绿色抗锯齿直线（起点(50,100)，终点(450,100)，线宽2，LINE_AA）  
    cv2.line(canvas, (50, 100), (450, 100), (0, 255, 0), 2, cv2.LINE_AA)  

    # 绘制蓝色4连通直线（起点(50,150)，终点(450,150)，线宽1，LINE_4）  
    cv2.line(canvas, (50, 150), (450, 150), (255, 0, 0), 1, cv2.LINE_4)  

    # 绘制带坐标偏移的直线（实际坐标为(100/2, 300/2)=(50,150)到(400/2,400/2)=(200,200)）  
    cv2.line(canvas, (100, 300), (400, 400), (100, 200, 250), 2, shift=1)  

    cv2.imshow("Line Demo", canvas)  
    cv2.waitKey(0)  
    cv2.destroyAllWindows()  
"""
