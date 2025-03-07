import cv2
import numpy as np
def test001():
    img=cv2.imread("./src/huofu.png")
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    h,w=img_gray.shape
    arr=cv2.HoughLines(img_gray,0.5,0.3*np.pi/180,100)
    print(arr)
    for el in arr:
        rho,theta=el[0]
        print(rho,theta,"==============")
        # x*np.cos(theta)+y*np.sin(theta)=rho
        if theta==0 or theta==np.pi:
            continue
        y=lambda x:-(x*np.cos(theta)-rho)/np.sin(theta)
        p1=(0,int(y(0)))
        p2=(w,int(y(w)))
        cv2.line(img,p1,p2,(0,0,255),1)
    cv2.imshow("img",img)
    cv2.waitKey(0)
# test001()

def test002():
    img=cv2.imread("./src/huofu.png")
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    h,w=img_gray.shape
    arr=cv2.HoughLinesP(img_gray,0.8, 0.01745, 90, minLineLength=120, maxLineGap=10)
    print(arr)
    for el in arr:
        x1,y1,x2,y2=el[0]
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
    cv2.imshow("img",img)
    cv2.waitKey(0)

test002()