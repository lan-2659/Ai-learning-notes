import cv2
import numpy as np

def test001():
    img=cv2.imread("./src/game.png")
    template=cv2.imread("./src/temp.png")
    h,w,bgr=template.shape

    res=cv2.matchTemplate(img,template,cv2.TM_SQDIFF)
    # res[res<100]

    print(res,cv2.minMaxLoc(res))
    min_num,max_num,min_loc,max_loc=cv2.minMaxLoc(res)
    clum,row=min_loc
    print(res[row,clum])
    cv2.rectangle(img,(clum,row),(clum+w,row+h),(0,0,255),2)

    cv2.imshow("img",img)
    cv2.waitKey(0)

def test002():
    img=cv2.imread("./src/game.png")
    template=cv2.imread("./src/temp.png")
    h,w,bgr=template.shape

    res=cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
    threshold=0.8
    indexs=np.where(res>=threshold)
    for y,x in zip(*indexs):
         cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow("img",img)
    cv2.waitKey(0)  
  

if __name__ == '__main__':
    # arr=[10,20,30,22,1]
    # arr2=[1,12,33,42,5]
    # re=zip(*[arr,arr2])
    # # print(re)
    # # it=iter(re)
    # # print(next(it))
    # # print(next(it))
    # for el in re:
    #     print(el)
    # arr=np.array([[1,2,3,4,5],
    #               [1,2,3,4,5],
    #               [10,20,30,40,50]])
    # arr2=np.where(arr>3)
    # print(arr2)
    # for el in zip(*arr2):
    #     print(el)
    # print(arr[np.int64(0),np.int64(0)])
    # test001()
    test002()