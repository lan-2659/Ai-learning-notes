
# cv2.getRotationMatrix2D()方法  
"""  
    语法格式：  
    M = cv2.getRotationMatrix2D(center, angle, scale)  

        参数说明：  
        center: 旋转中心坐标（图像坐标系下的坐标点），格式为 (x, y)（整数或浮点数）  
            图像坐标系原点在左上角，x轴向右，y轴向下  
            通常设为图像中心：(cols//2, rows//2)，其中cols、rows为图像宽高  
        angle: 旋转角度（单位：度/°）  
            正值：逆时针旋转  
            负值：顺时针旋转  
        scale: 缩放因子（非负浮点数）  
            1.0：保持原尺寸  
            0.5：缩小为原尺寸的50%  
            2.0：放大为原尺寸的200%  


    返回值：  
    M: 2×3的仿射变换矩阵（数据类型为np.float32）


    注意事项：  
    1. 旋转中心坐标 `center` 是图像坐标系下的绝对坐标，而非相对坐标  
    2. 旋转矩阵包含 **旋转** 和 **平移** 两部分：  
        - 平移部分用于将旋转中心从原点调整到指定的 `center` 位置  
        - 直接使用该矩阵进行变换时，无需额外构造平移矩阵  
    3. 矩阵数据类型固定为 `np.float32`，可直接传入 `cv2.warpAffine` 使用  
    4. 角度单位为 **度**（°），而非弧度（rad），内部会自动转换（使用 `np.deg2rad(angle)`）  
    5. 缩放因子 `scale` 可实现旋转时的同步缩放（如 `scale=0.8` 表示旋转同时缩小20%）   
"""


# cv2.warpAffine()方法  
"""  
    语法格式：  
    cv2.warpAffine(src, M, dsize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)  

        参数说明：  
        src: 输入图像（numpy数组，单通道或三通道图像，数据类型为uint8或float32）  
        M: 2×3仿射变换矩阵（数据类型必须为np.float32），控制图像的线性变换和平移  
        dsize: 输出图像尺寸（宽度, 高度），格式为元组(int, int)，需根据变换调整避免图像截断  
        flags: 插值方法（可选参数，默认值为cv2.INTER_LINEAR）  
            常用选项：  
                cv2.INTER_NEAREST：最近邻插值（速度快，适合像素化图像）  
                cv2.INTER_LINEAR：双线性插值（默认，平衡速度与质量）  
                cv2.INTER_AREA：区域插值（缩小图像时抗锯齿效果好）  
                cv2.INTER_CUBIC：双三次插值（高质量，计算较慢）  
                cv2.INTER_LANCZOS4：Lanczos插值（最高质量，计算量最大）  
        borderMode: 边界填充模式（可选参数，默认值为cv2.BORDER_CONSTANT）  
            常用选项：  
                cv2.BORDER_CONSTANT：用固定值填充（由borderValue指定，默认黑色）  
                cv2.BORDER_REPLICATE：复制边缘像素（如边缘为[10, 20], 填充为[10, 10, 20, 20]）  
                cv2.BORDER_REFLECT：镜像反射填充（如边缘为[10, 20], 填充为[20, 10, 20, 10]）  
                cv2.BORDER_WRAP：循环填充（超出部分回绕到对侧，可能产生错位）  
        borderValue: 边界填充值（可选参数，默认值为0）  
            单通道图像：标量值（如0表示黑色，255表示白色）  
            三通道图像：BGR格式元组（如(255, 0, 0)表示蓝色，(0, 255, 0)表示绿色）  


    注意事项：  
    1. 返回值：返回变换后的图像（ndarray类型），不修改输入图像src  
    2. 变换矩阵M必须为np.float32类型，否则会抛出断言错误（Assertion failed）  
    3. 输出尺寸dsize需根据变换类型调整：  
        - 平移后：通常设为原尺寸+平移量，避免图像被裁剪  
        - 旋转后：需通过几何计算（如旋转矩阵的余弦/正弦值）计算最小外接矩形尺寸  
    4. 插值方法选择：放大图像时优先用cv2.INTER_CUBIC/LINEAR，缩小图像时优先用cv2.INTER_AREA  
    5. 组合变换：复杂变换可通过矩阵乘法组合（如M_combined = M_translate @ M_rotate），注意矩阵乘法顺序（右乘顺序为变换执行顺序）  


    举例：  
    import cv2  
    import numpy as np  

    # 示例1：图像旋转（绕中心逆时针旋转45°，保持原尺寸）  
    img = cv2.imread("lena.jpg", cv2.IMREAD_COLOR)  
    rows, cols = img.shape[:2]  
    center = (cols // 2, rows // 2)  
    angle = 45  
    scale = 1.0  
    M = cv2.getRotationMatrix2D(center, angle, scale)  # 生成旋转矩阵  
    rotated_img = cv2.warpAffine(img, M, (cols, rows))  # 不扩展尺寸，边缘用默认黑色填充  

    # 示例2：图像平移（向右平移100px，向下平移50px，扩展输出尺寸）  
    tx, ty = 100, 50  
    M_translate = np.float32([[1, 0, tx], [0, 1, ty]])  
    translated_img = cv2.warpAffine(img, M_translate, (cols + tx, rows + ty), borderValue=(255, 255, 255))  # 白色边界填充  

    # 示例3：图像缩放（宽高各缩小50%，以图像中心为锚点）  
    sx, sy = 0.5, 0.5  
    anchor_x, anchor_y = cols // 2, rows // 2  
    M_scale = np.float32([[sx, 0, anchor_x * (1 - sx)], [0, sy, anchor_y * (1 - sy)]])  # 带锚点的缩放矩阵  
    scaled_img = cv2.warpAffine(img, M_scale, (int(cols * sx), int(rows * sy)), flags=cv2.INTER_AREA)  # 缩小图像用区域插值  

    # 示例4：图像剪切（沿x轴剪切，剪切角30°）  
    shear_factor = np.tan(np.deg2rad(30))  # 30°剪切角对应的tan值  
    M_shear = np.float32([[1, shear_factor, 0], [0, 1, 0]])  
    sheared_img = cv2.warpAffine(img, M_shear, (cols, rows), borderMode=cv2.BORDER_REPLICATE)  # 复制边缘像素避免黑边  

    # 示例5：组合变换（先缩放后旋转，最后平移）  
    M_scale = np.float32([[0.8, 0, 0], [0, 0.8, 0]])  # 缩放矩阵  
    M_rotate = cv2.getRotationMatrix2D(center, 30, 1.0)  # 旋转矩阵  
    M_translate = np.float32([[1, 0, 50], [0, 1, 50]])  # 平移矩阵  
    M_combined = M_translate @ M_rotate @ M_scale  # 矩阵相乘顺序：平移×旋转×缩放（从右到左执行）  
    combined_img = cv2.warpAffine(img, M_combined, (cols + 50, rows + 50), flags=cv2.INTER_LINEAR)  

    cv2.imshow("WarpAffine Demo", combined_img)  
    cv2.waitKey(0)  
    cv2.destroyAllWindows()  
"""