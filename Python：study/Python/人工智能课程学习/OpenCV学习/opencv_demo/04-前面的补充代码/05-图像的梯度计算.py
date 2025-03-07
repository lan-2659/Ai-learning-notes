import cv2
import numpy as np
def test001():
    img=np.array([[100,101,103,99,0,0,0,209,204,205],
                  [100,101,103,99,0,0,0,209,204,205],
                  [100,101,103,99,0,0,0,209,204,205],
                  [100,101,103,99,0,0,0,209,204,205],
                  [100,101,103,99,0,0,0,209,204,205],
                  [100,101,103,99,0,0,0,209,204,205],
                  [100,101,103,99,0,0,0,209,204,205]],dtype=np.uint8)
    k=np.array([[-1,0,1],
                [-2,0,2],
                [-1,0,1]])
    img2=cv2.filter2D(img,-1,k)
    print(img2)
# test001()

def test002():
    img=cv2.imread("./src/shudu.png",cv2.IMREAD_GRAYSCALE)
    k=np.array([ [-1,0,1],
                 [-2,0,2],
                 [-1,0,1]])
    img2=cv2.filter2D(img,-1,kernel=k)
    img3=cv2.filter2D(img,-1,kernel=k.T)
    img5=cv2.Sobel(img,-1,1,1,ksize=3)
    img6=cv2.Laplacian(img,-1,ksize=3)
    cv2.imshow("img",img)
    cv2.imshow("img2",img2)
    cv2.imshow("img3",img3)
    cv2.imshow("img4",img2+img3)
    cv2.imshow("img5",img5)
    cv2.imshow("img6",img6)
    cv2.waitKey(0)
test002()