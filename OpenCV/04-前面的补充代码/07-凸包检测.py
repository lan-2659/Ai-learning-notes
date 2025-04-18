import cv2
def test001():
    img=cv2.imread("./src/num.png",)
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 找出图像的轮廓
    contours,_=cv2.findContours(img_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    cv2.drawContours(img, contours, 2, (255,0,0), 2)

    # 轮廓的凸包点检测
    hull=cv2.convexHull(contours[1])
    hull2=cv2.convexHull(contours[2])
    # print(hull)
    # cv2.drawContours(img,[hull,hull2],0,(0,255,0),2)
    cv2.polylines(img, [hull,hull2], isClosed=True,color=(0,0,255), thickness=2)

    cv2.imshow("img",img)
    cv2.waitKey(0)
test001()