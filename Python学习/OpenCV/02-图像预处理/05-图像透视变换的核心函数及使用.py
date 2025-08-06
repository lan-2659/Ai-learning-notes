# 创建透视变换矩阵
"""  
# cv2.getPerspectiveTransform()方法  

    语法格式：  
    M = cv2.getPerspectiveTransform(src, dst)  

        参数说明：  
        src: 源图像中四个点的坐标，需要输入一个数组，形状为 (4, 2)，表示四个点的 (x, y) 坐标。
        dst: 目标图像中对应的四个点的坐标，同样需要输入一个数组，形状为 (4, 2)。

    返回值：  
    M: 3×3 的透视变换矩阵（数据类型为 `np.float32`）

    注意事项：  
    1. 源图像和目标图像中的四个点必须一一对应，且这四个点不能共线，否则无法计算透视变换矩阵。
    2. 矩阵数据类型固定为 `np.float32`，可直接传入 `cv2.warpPerspective` 使用。
    3. 该方法通过这四个对应点，利用最小二乘法等数学方法计算出能够将源图像映射到目标图像的透视变换矩阵。
"""

# 使用透视变换矩阵
"""  
# cv2.warpPerspective()方法  

    语法格式：  
    cv2.warpPerspective(src, M, dsize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)  

        参数说明：  
        src: 输入图像
        M: 3×3 透视变换矩阵（数据类型必须为 `np.float32`）
        dsize: 输出图像尺寸（宽度, 高度），格式为元组 `(int, int)`，需根据变换调整避免图像截断。
        flags: 插值方法（可选参数，默认值为 `cv2.INTER_LINEAR`）  
            常用选项：  
                cv2.INTER_NEAREST：最近邻插值（速度快，适合像素化图像）  
                cv2.INTER_LINEAR：双线性插值（默认，平衡速度与质量）  
                cv2.INTER_AREA：区域插值（缩小图像时抗锯齿效果好）  
                cv2.INTER_CUBIC：双三次插值（高质量，计算较慢）  
                cv2.INTER_LANCZOS4：Lanczos插值（最高质量，计算量最大）  
        borderMode: 边界填充模式（可选参数，默认值为 `cv2.BORDER_CONSTANT`）  
            常用选项：  
                cv2.BORDER_CONSTANT：用固定值填充（由 `borderValue` 指定，默认黑色）  
                cv2.BORDER_REPLICATE：复制边缘像素（如边缘为 `[10, 20]`, 填充为 `[10, 10, 20, 20]`）  
                cv2.BORDER_REFLECT：镜像反射填充（如边缘为 `[10, 20]`, 填充为 `[20, 10, 20, 10]`）  
                cv2.BORDER_WRAP：循环填充（超出部分回绕到对侧，可能产生错位）  
        borderValue: 边界填充值（可选参数，默认值为 0）  
            单通道图像：标量值（如 0 表示黑色，255 表示白色）  
            三通道图像：BGR 格式元组（如 `(255, 0, 0)` 表示蓝色，`(0, 255, 0)` 表示绿色）

    注意事项：  
    1. 返回值：返回变换后的图像（`ndarray` 类型），不修改输入图像 `src`。
    2. 变换矩阵 `M` 必须为 `np.float32` 类型，否则会抛出断言错误（Assertion failed）。
    3. 输出尺寸 `dsize` 需根据变换类型调整，可能需要通过几何计算来确定合适的尺寸，以避免图像被裁剪。
    4. 插值方法选择：放大图像时优先用 `cv2.INTER_CUBIC` 或 `cv2.INTER_LINEAR`，缩小图像时优先用 `cv2.INTER_AREA`。
    5. 透视变换常用于图像校正、鸟瞰图转换等场景，能够将图像从一个透视视角转换到另一个透视视角。

    举例：  
    import cv2  
    import numpy as np  

    # 读取图片
    img = cv2.imread("images/1.jpg", cv2.IMREAD_COLOR)  

    # 源图像中的四个点
    src = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])

    # 目标图像中对应的四个点
    dst = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src, dst)

    # 应用透视变换
    warped_img = cv2.warpPerspective(img, M, (300, 300))

    cv2.imshow("Perspective Warp", warped_img)  
    cv2.waitKey(0)  
    cv2.destroyAllWindows()  
"""