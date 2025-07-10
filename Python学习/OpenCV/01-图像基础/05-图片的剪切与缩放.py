# 图片的剪切
"""
由于在OpenCv中，会将图片读取为 ndarray对象
因此，对图片的裁剪其实就是进行数组的切片


举例：
import cv2

image = cv2.imread("../../images/car3.png")     # 读取图片为 ndarray对象

x, y, w, h = 0, 0, 200, 400

height, width, c = image.shape      # 获取图片的宽度，高度
print(height, width, c)


if y+h < height and x+w < width:
   
    img = image[y:y+h, x:x+w]       # 裁剪（数组切片）

    cv2.imshow("image01", image)
    cv2.imshow("image02", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("索引越界")
"""


# 图片的缩放
"""
cv2.resize()方法：
    语法格式：  
    cv2.resize(src, dsize, dst=None, fx=0, fy=0, interpolation=INTER_LINEAR)  

        参数说明：  
        src: 输入图像（numpy数组）  
        dsize: 输出图像尺寸，格式为 (width, height)  
        dst: 输出图像（可选），传入一个预先创建好的数组，函数会把结果填充进这个数组中(不返回值)，在需要将图片频繁缩放到同一尺寸的场景下可以显著提升性能
        fx: 水平轴缩放比例因子 
        fy: 垂直轴缩放比例因子  
        interpolation: 插值方法，默认为 cv2.INTER_LINEAR  

    注意事项：
    返回一个新的图像数组  
    尺寸优先级：  
        若指定了dsize，则忽略fx和fy  
        若dsize为(0,0)或None，则通过fx和fy计算目标尺寸  
    插值方法（常用）：  
        cv2.INTER_NEAREST: 最近邻插值（速度快，质量差）  
        cv2.INTER_LINEAR: 双线性插值（默认，平衡速度质量）  
        cv2.INTER_CUBIC: 双三次插值（高质量，速度慢）  
        cv2.INTER_AREA: 区域插值（图像缩小时推荐）  
    尺寸陷阱：  
        输入dsize顺序是 (宽度, 高度)  
        与numpy数组的shape (高度, 宽度, 通道) 顺序相反  
  

    举例：  
    import cv2

    img = cv2.imread("../images/car3.png")      # 读取图片为 ndarray对象

    resized1 = cv2.resize(img, (300, 200))      # 指定目标尺寸 
      
    resized2 = cv2.resize(img, None, fx=2, fy=1.5)          # 使用缩放因子（宽度x2，高度x1.5）
      
    resized3 = cv2.resize(img, (400, 300), fx=0.5, fy=0.5)  # 组合使用（最终尺寸=dsize，忽略fx/fy）

    cv2.imshow("image", img)
    cv2.imshow("resized1", resized1)
    cv2.imshow("resized2", resized2)
    cv2.imshow("resized3", resized3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""


