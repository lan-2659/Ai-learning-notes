# OpenCv中坐标(x, y)
"""
在OpenCv中图像是以 numpy.ndarray 的形式存储的
如下图所示：（'*'代表像素）

    0 1 2 3 4 5 6 7 8 9 ————-————-————-————-> x
 0  * * * * * * * * * * * * * * * * * * * *
 1  * * * * * * * * * * * * * * * * * * * *
 2  * * * * * * * * * * * * * * * * * * * *
 3  * * * * * * * * * * * * * * * * * * * *
 4  * * * * * * * * * * * * * * * * * * * *
 5  * * * * * * * * * * * * * * * * * * * *
 6  * * * * * * * * * * * * * * * * * * * *
 7  * * * * * * * * * * * * * * * * * * * *
 8  * * * * * * * * * * * * * * * * * * * *
 9  * * * * * * * * * * * * * * * * * * * *
 |  * * * * * * * * * * * * * * * * * * * *
 |  * * * * * * * * * * * * * * * * * * * *
 |  * * * * * * * * * * * * * * * * * * * *
 |  * * * * * * * * * * * * * * * * * * * *
 V   
 y   

最左上角的'*'坐标为(0, 0)
从左上角开始，x向左为正方向，y向下为正方向

向OpenCv中的方法传递(x, y)坐标时，(x, y)坐标以上述方法为准
"""


# 最近邻插值(CV2.INTER_NEAREST)
"""
优势领域：
    快速缩放(低质量需求)

1. 核心概念
最近邻插值是图像处理中最简单、计算速度最快的插值方法。
其核心思想是：当需要生成新像素时，直接取原图像中距离映射点最近的像素值。
它不涉及复杂的数学计算，仅依赖位置关系，因此效率极高，但可能会导致图像边缘出现锯齿

2. 数学原理
    设原图像尺寸为 W:H, 缩放后的目标图像尺寸为 W′:H′ 则缩放比例为：
        scale_x = W / W′
        scale_y = H / H′
        
    目标图像中任意像素点 (x′, y′) 对应的原图像坐标 (x, y) 计算方式为：
        x = round(x′ * scale_x)
        y = round(y′ * scale_y)
"""


# 双线性插值(CV2.INTER_LINEAR)
"""
优势领域：
    中等质量缩放，可用于既要放大又要缩小的地方(OpenCV的默认插值方法，计算速度不是很慢，质量也不是很差)

1. 核心概念
双线性插值是OpenCV图像处理中最常用的插值方法，计算速度比最近邻插值要慢，但处理结果比最近邻插值好。
双线性插值是一种基于映射点周围四个最近像素值的图像插值方法，通过两次线性插值（水平方向和垂直方向）计算新像素值。
相较于最近邻插值，它能生成更平滑的图像，减少锯齿现象，但计算复杂度稍高。

2. 数学原理
    设原图像尺寸为 W:H, 缩放后的目标图像尺寸为 W′:H′ 则缩放比例为：
        scale_x = W / W′
        scale_y = H / H′
        
    目标图像中任意像素点 Q′(x′, y′) 对应的原图像浮点坐标坐标 Q(x, y) 计算方式为：
        x = x′ * scale_x
        y = y′ * scale_y
    
    取浮点坐标 Q(x, y) 周围四个像素点进行加权平均(按距离算权重，距离越近权重越大)(两次水平加权，一次垂直加权):
        i = int(x)
        j = int(y)

        Q11 = (i, j)
        Q12 = (i, j+1)
        Q21 = (i+1, j)
        Q22 = (i+1, j+1)

        则这四个点与浮点坐标 Q(x, y) 的相对位置如下：

        (i, j)    x1    Q1     x2         (i+1, j)  
            Q11--------*----------------Q12
                       | y1
                       |
                       Q(x, y)
                       |
                       | 
                       | y2
                       |
                       |
                       |
            Q21--------*----------------Q22
      (i, j+1)          Q2                (i+1, j+1)

        Q1 = (x2 * Q11 + x1 * Q12) / (x2 + x1)
        Q2 = (x2 * Q21 + x1 * Q22) / (x2 + x1)

        Q = (y2 * Q1 + y1 * Q2) / (y2 + y1)

        注: x2 + x1 和 y2 + y1 的值为 1 其实可以直接省去，但是怕以后忘记，所以保留
"""


# 区域插值(cv2.INTER_AREA)
"""
优势领域：
    图像缩小(计算速度比双线性插值快一点)

1、核心概念
这是一种专门针对图像缩小（降采样）设计的插值算法。
其核心原理是通过计算目标像素对应原图像区域的加权平均

2、数学原理
    设原图尺寸 W:H，目标图尺寸 W':H'，则缩放比例：
        scale_x = W / W'
        scale_y = H / H'

    目标图坐标 Q'(x', y') 映射到原图浮点坐标：
        x = x' * scale_x
        y = y' * scale_y

    在水平方向上取区间 [x, x+scale_x] (决定所取像素的x坐标)
    在垂直方向上取区间 [y, y+scale_y] (决定所取像素的y坐标)
    由这两个区间取出对应的像素点，然后进行加权平均，得到新像素值

    每个取到的像素的权重为其水平方向重叠长度与垂直方向重叠长度的乘积
        如水平坐标范围是 [0, 1.5]，垂直坐标范围是 [0, 1.5]
        则像素 (0, 1)：
            这个像素在水平方向上的长度为 [0, 1) 这个区间，垂直方向上的长度为 [1, 2) 这个区间
            水平重叠长度为 1 
            垂直重叠长度为 0.5
            覆盖面积(权重) A(0,1)=1×0.5=0.5

    新像素的值 = 所有像素与其权重相乘的和 / 所有像素的权重之和
"""


# 双三次插值(cv2.INTER_CUBIC)
"""
优势领域：
    图像放大(计算速度比双线性插值要慢，质量较高)

1. 核心概念
双三次插值计算速度比双线性插值还要慢，但处理结果比双线性插值好。           
双三次插值通过原图像中映射点周围 16个邻域像素 和 权重函数(Catmull-Rom函数 、BiCubic函数) 计算每个像素权重
将16个邻域像素加权求和，得到新像素值，实现更平滑的图像缩放或变换

2. 数学原理
    设原图尺寸 W:H，目标图尺寸 W':H'，则缩放比例：
        scale_x = W / W'
        scale_y = H / H'

    目标图坐标 Q'(x', y') 映射到原图浮点坐标：
        x = x' * scale_x
        y = y' * scale_y

    需要取原图坐标周围 4x4=16 个邻域像素，通过权重函数计算权重：
    权重函数 (Catmull-Rom)：
        W(t) = 
            1.5|t|³ - 2.5|t|² + 1           , |t| ≤ 1
            -0.5|t|³ + 2.5|t|² -4|t| + 2    , 1 < |t| < 2
            0                                , 其他
    注：t为x方向上的水平距离或y方向上的水平距离
    
    每个领域像素都有两个权重，一个x方向，一个y方向

    新像素 = Σ 领域像素 * 权重(x方向) * 权重(y方向)
"""


# Lanczos插值 (cv2.INTER_LANCZOS4)——(cv2.INTER_LANCZOS4这个是四阶的，a=4)
"""
优势领域：
    专业级缩放，保留所有细节和锐度(计算速度比双三次插值要慢，质量较高)

1、核心概念
lanczos插值一般来说比双三次插值性能更好，但计算速度比双三次插值慢
Lanczos插值是一种高质量图像重采样方法，尤其适用于图像缩放（放大或缩小）
它通过Lanczos核函数对周围像素进行加权计算，能在保留高频细节的同时减少锯齿和模糊

2、数学原理
    设原图尺寸 W:H，目标图尺寸 W':H'，则缩放比例：
        scale_x = W / W'
        scale_y = H / H'

    目标图坐标 Q'(x', y') 映射到原图浮点坐标：
        x = x' * scale_x
        y = y' * scale_y

    需要取原图坐标周围 2a x 2a 个邻域像素，通过权重函数计算权重：
    Lanczos核函数：
        L(x)=
            sinc(x)⋅sinc(x/a)  , |x| < a
            0                 , |x| >= a 
    注：
        x 为领域像素到映射像素的水平或垂直距离距离
        OpenCv中默认a=3(6x6邻域)
    
    每个领域像素都有两个权重，一个x方向，一个y方向

    新像素 = Σ 领域像素 * 权重(x方向) * 权重(y方向)
"""
