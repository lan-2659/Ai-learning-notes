"""
图像的直方图统计并绘制
"""
import cv2
import numpy as np

def showhist(windowname,img):
    # 统计直方图数据
    # [img]表示 可以传入多个图像
    hist=cv2.calcHist([img],[0],None,[256],[0,256])
    # print(hist)
    re=cv2.minMaxLoc(hist)
    print(re)
    d=400/re[1]

    # 创建一个图(用于直方图绘制)
    img_hist=np.zeros((512,712,3),dtype=np.uint8)
    
    # 绘制直方图
    for i in range(len(hist)):
        x=20+i*2
        y1=500
        y2=500-int(20+hist[i][0]*d)
        cv2.line(img_hist,(x,y1),(x,y2),(0,0,255),2,lineType=cv2.LINE_AA)
    cv2.imshow(windowname,img_hist)
def test001():
    img=cv2.imread("./src/zhifang.png",cv2.IMREAD_GRAYSCALE)
    showhist(img)
    cv2.imshow("img",img)
    cv2.waitKey(0)
# test001()


    
def test002():
    img=cv2.imread("./src/zhifang.png",cv2.IMREAD_GRAYSCALE)
    img2=cv2.equalizeHist(img)#自适应直方图均衡化
    cla=cv2.createCLAHE(2,(8,8))
    img3=cla.apply(img)#对比度受限的自适应直方图均衡化

    showhist("img_hist",img)
    showhist("img2_hist",img2)
    showhist("img3_hist",img3)
    cv2.imshow("img2",img2)
    cv2.imshow("img",img)
    cv2.imshow("img3",img3)
    cv2.waitKey(0)
test002()