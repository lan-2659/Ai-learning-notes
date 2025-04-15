# 圆形绘制
"""
cv2.circle()方法：
    语法格式：
    cv2.circle(img, center, radius, color, thickness=None, lineType=None, shift=None)

        参数说明：
        img: 输入图像（numpy数组，在该图像上绘制圆）
        center: 圆心坐标，格式为 (x, y)
        radius: 圆的半径（整数）
        color: 圆的颜色，BGR格式（例如 (255,0,0) 表示蓝色）
        thickness: 线条粗细（默认1）。若为 -1 表示填充圆形
        lineType: 线条类型（默认cv2.LINE_8）
        shift: 坐标小数点位数（默认为0，表示整数坐标）

    注意事项：
    直接在原图上修改，无返回值

    举例：
    import cv2
    import numpy as np

    # 创建512x512的黑色画布
    canvas = np.zeros((512,512,3), dtype=np.uint8)

    # 绘制红色空心圆（中心(200,200)，半径50，线宽3）
    cv2.circle(canvas, (200,200), 50, (0,0,255), 3)

    # 绘制绿色实心圆（中心(400,400)，半径30）
    cv2.circle(canvas, (400,400), 30, (0,255,0), -1)

    # 绘制带小数坐标的圆（使用shift参数）
    cv2.circle(canvas, (300,300), 40, (255,0,0), 2, shift=2)  # 实际坐标为(300/4, 300/4)

    cv2.imshow("Circles Demo", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""
