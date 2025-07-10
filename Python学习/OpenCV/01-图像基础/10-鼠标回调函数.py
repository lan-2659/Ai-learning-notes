# cv2.setMouseCallback()方法  
"""  
    语法格式：  
    cv2.setMouseCallback(windowName, onMouse, userData=None)  

        参数说明：  
        windowName: 目标窗口名称（这个窗口需是提前创建好的）  
        onMouse: 鼠标事件回调函数 (这个函数需要自己定义)
        userData: （可选）传递给回调函数的用户数据  

    注意事项：  
    无返回值，cv2.setMouseCallback方法的作用是，检测用户对指定窗口的鼠标事件和鼠标位置，并将这些信息传递给回调函数
    onMouse：(这个函数的定义格式是固定的)

        语法格式：(这个语法格式中只有函数名可以更改)
        def onMouse (event, x, y, flags, param):
        
            参数说明：  
            event：鼠标事件类型(下面的是几个常用的)  
                cv2.EVENT_LBUTTONDOWN：左键按下（数值 1）
                cv2.EVENT_LBUTTONUP：左键释放（数值 4）
                cv2.EVENT_MOUSEMOVE：鼠标移动（数值 0）
                cv2.EVENT_RBUTTONDOWN：右键按下（数值 2）
            x：鼠标当前位置的x坐标(会随着鼠标的移动而变化)  
            y：鼠标当前位置的y坐标(会随着鼠标的移动而变化)
            flags：鼠标事件标志  
            param：用户自定义参数
        

    举例：点击图像绘制蓝色圆点  
    import cv2  
    import numpy as np  

    img = np.zeros((512, 512, 3), dtype=np.uint8)  
    cv2.namedWindow("Image")  

    def click_event(event, x, y, flags, param):  
        if event == cv2.EVENT_LBUTTONDOWN:  
            cv2.circle(img, (x, y), 5, (255, 0, 0), -1)  # 绘制实心圆  
            cv2.imshow("Image", img)  

    cv2.setMouseCallback("Image", click_event)  

    while True:  
        cv2.imshow("Image", img)  
        if cv2.waitKey(1) == 27:  # ESC退出  
            break  

    cv2.destroyAllWindows()  
"""