
# 图片翻转
"""  
# cv2.flip()方法  

    语法格式：  
    cv2.flip(src, flipCode, dst=None)  

        参数说明：  
        src: 输入图像（必需参数）  
            ndarray：单通道或三通道图像（灰度图/彩色图），数据类型通常为np.uint8  
        flipCode: 翻转方向控制（必需参数）  
            int类型: 
                0: 垂直翻转(沿X轴的图像中线 y=H/2)      像素(x, y)变为(W-1-x, y)
                1: 水平翻转(沿Y轴的图像中线 x=W/2)      像素(x, y)变为(x, H-1-y) 
                -1: 双向翻转(将上述两个翻转各进行一次)   像素(x, y)变为(W-1-x, H-1-y)
        dst: 输出图像（可选参数）  
            ndarray：与src形状、数据类型相同的翻转后图像，若不指定则自动创建  


    注意事项：  
    返回反转后的图像   


    举例：  
    import cv2  
    import numpy as np  

    # 场景1：基础翻转操作（读取中文路径图像，复用字节流方案）  
    def read_image(path):  
        return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)  

    image = read_image("图片/风景.jpg")  # 假设为512x512彩色图  

    # 水平翻转（左右镜像）  
    flipped_h = cv2.flip(image, 1)  
    # 垂直翻转（上下颠倒）  
    flipped_v = cv2.flip(image, 0)  
    # 双向翻转（完全镜像）  
    flipped_hv = cv2.flip(image, -1)  

    # 场景2：视频流实时翻转 
    cap = cv2.VideoCapture(0) 

    while cap.isOpened():  
        ret, frame = cap.read()  
        if not ret: 
            break  
        corrected_frame = cv2.flip(frame, 1)  # 前置摄像头画面水平翻转 
        cv2.imshow("Corrected Camera", corrected_frame)  
        if cv2.waitKey(1) == ord('q'):  
            break  
    cap.release()  
    cv2.destroyAllWindows()  
"""