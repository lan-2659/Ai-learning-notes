import cv2  
import numpy as np  

# 创建512x512的黑色画布（BGR三通道）  
canvas = np.zeros((512, 512, 3), dtype=np.uint8)  

# 绘制红色直线（起点(50,50)，终点(450,50)，线宽3，默认LINE_8）  
cv2.line(canvas, (50, 50), (450, 50), (0, 0, 255), 0)  


cv2.imshow("Line Demo", canvas)  
cv2.waitKey(0)  
cv2.destroyAllWindows()