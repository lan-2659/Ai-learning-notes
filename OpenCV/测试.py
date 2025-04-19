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