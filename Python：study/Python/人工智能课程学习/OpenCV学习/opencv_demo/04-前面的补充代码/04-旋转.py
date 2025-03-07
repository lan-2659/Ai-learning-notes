import cv2
import numpy as np


def test001():
    img = np.fromfile(r'D:\Python：study\Python\人工智能课程学习\OpenCV学习\opencv_demo\04-前面的补充代码\src\face.png', dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    print(img.shape)
    h, w, bgr = img.shape
    # deg=np.pi/180
    m = cv2.getRotationMatrix2D((h // 2, w // 2), 0, 2)
    img2 = cv2.warpAffine(img, m, (2 * h, 2 * w))
    cv2.imshow("img2", img2)
    cv2.imshow("img", img)
    cv2.imwrite("./src/face2.png", img2)
    cv2.waitKey(0)


# test001()


def test002():
    img = cv2.imread("./src/face.png")
    print(img.shape)
    h, w, bgr = img.shape
    # deg=np.pi/180
    m = cv2.getRotationMatrix2D((h // 2, w // 2), 0, 2)
    img2 = cv2.warpAffine(
        img,
        m,
        (2 * h, 2 * w),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    cv2.imshow("img2", img2)
    cv2.imshow("img", img)
    cv2.imwrite("./src/face2.png", img2)
    cv2.waitKey(0)


test002()
