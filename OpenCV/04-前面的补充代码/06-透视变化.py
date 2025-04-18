import cv2
import numpy as np
def test001():
    img=cv2.imread("./src/3.png")
    h,w,bgr=img.shape
    pts1 = np.float32([[178, 100], [487, 134], [124, 267], [473, 308]])
    pts2 = np.float32([[0, 0], [w, 0],[0, h], [w, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img2=cv2.warpPerspective(img,M,(w,h))
    cv2.imshow("img2",img2)
    cv2.imshow("img",img)

    cv2.waitKey(0)
test001()