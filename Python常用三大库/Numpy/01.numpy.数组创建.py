# NumPy 定义了一个 n 维数组对象，简称 ndarray 对象


# ndarray对象 是一个由一系列相同类型元素组成的数组集合
# 数组中的每个元素都占有大小相同的内存块，且这些内存块是连续的


# 除 0 维数组（标量数组）外，其他维度的 ndarray 均可迭代
"""
当标量与 ndarray 进行运算时，标量会被隐式视为 0 维数组，并通过广播机制扩展为与数组相同的形状
"""


# 创建 ndarray对象 时，order参数决定了数组中每个元素的具体存储位置：
"""
以形状为 (2, 3) 的数组为例:
# C-按行存储：
    将逻辑结构上位于这些位置的元素依次填入连续的内存块
    (0, 0)  (0, 1)  (0, 2)  (1, 0)  (1, 1)  (1, 2)
    
# F-按列存储：
    将逻辑结构上位于这些位置的元素依次填入连续的内存块
    (0, 0)  (1, 0)  (0, 1)  (1, 1)  (0, 2)  (1, 2)

实际上就是从逻辑位置 (0, 0) 开始每次加1，然后根据位置来存储值；如果 order='C' 就从右边开始加 ，如果 order='F' 就从左边加
"""


# 创建 ndarray对象 的方法
"""
import numpy as np


# array() 方法：

    语法格式：
    numpy.array(object, dtype=None, copy=True, order='C', ndmin=0)  

        这个方法会返回一个 ndarray对象，返回对象的逻辑结构由object参数决定

        # 参数说明：
        object: 可以是任何可转换为数组的对象(如列表、元组、其它ndarray对象等)(并非是可迭代对象，如字符串、集合、字典等，这些会创建失败但不报错)
        dtype: 指定数组的数据类型，可选，如果不指定，则根据实际输入的数据类型自动判断
        copy: 默认为True，表示新数组中的数据与原数据是否完全独立（如果为False，则新数组与原数组共享同一批数据；如果为True，则会创建原数据的副本作为新数组的数据）
        order: 可以指定元素在内存中的存储方式(但不会影响逻辑结构)，C-按行，F-按列，默认为 C
        ndmin: 指定最小维度的参数，确保生成的数组至少具有指定的维度
        
        # 注意事项：
        copy在绝大多数情况下都是True（不管有没有指定copy为False），只有当object参数为ndarray对象，且dtype和order都不指定（或指定为和object相同的类型）时，copy=False才生效

    举例：
    arr = np.array([1, 2], dtype=int, copy=True, order='F', ndmin=0) 


# zeros() 方法：

    语法格式：
    numpy.zeros(shape, dtype=float, order='C')

        这个方法会返回一个全0填充的ndarray对象

        # 参数说明：
        shape: 指定数组的维度结构(一般是传入一个元组，如果是创建一维数组，则传入一个整数即可)
        dtype: 指定数组的数据类型，默认为float类型
        order: 指定元素在内存中的存储方式，C-按行优先，F-按列优先，默认为C

    举例：
    a = np.zeros((2, 3), dtype=int, order='C')

    
# ones() 方法

    语法格式：
    numpy.ones(shape, dtype=float, order='C')

        这个方法会返回一个全1填充的ndarray对象

        # 参数说明：
        shape: 指定数组的维度结构(一般是传入一个元组，如果是创建一维数组，则传入一个整数即可)
        dtype: 指定数组的数据类型，默认为float类型
        order: 指定元素在内存中的存储方式，C-按行优先，F-按列优先，默认为C

    举例：
    a = np.ones((4, 3), dtype=np.float32)

    
# full() 方法:

    语法格式：
    numpy.full(shape, fill_value, dtype=None, order='C')

        这个方法会返回一个由指定值填充的ndarray对象

        # 参数说明：
        shape: 指定数组的维度结构(一般是传入一个元组，如果是创建一维数组，则传入一个整数即可)
        fill_value: 指定数组的填充值（必须传入）
        dtype: 指定数组的数据类型（默认根据fill_value自动推断）
        order: 指定元素在内存中的存储方式，C-按行优先，F-按列优先，默认为C

    举例：
    a = np.full((3, 3), fill_value=5, dtype=np.uint8)

    
# arange() 方法:

    语法格式：
    numpy.arange(start=0, stop, step=1, dtype=None)

        这个方法会返回一维数组(数组中的元素是一个等差数列)

        # 参数说明：
        start: 起始值（默认0）
        stop: 结束值（结果不包含该值）
        step: 步长（默认1）
        dtype: 指定数组的数据类型（默认根据输入参数自动推断）
        
        # 注意事项：
        参数使用类似Python的range()函数，但是不同的是这个函数可以接收浮点数
        step不能为0

    举例：
    a = np.arange(0, 10, 1)  
    # 输出：[0 1 2 3 4 5 6 7 8 9]

    a = np.arange(10)  
    # 输出：[0 1 2 3 4 5 6 7 8 9]

    
# linspace() 方法:

    语法格式：
    numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)

        这个方法会返回一维数组(数组中的元素是在 [start, stop] 区间等距取的 num 个数字，start(第一个)和stop(最后一个)都包含在内)

        # 参数说明：
        start: 必须指定的起始值
        stop: 必须指定的结束值
        num: 生成的样本数量（默认50）
        endpoint: 是否包含stop值（默认True）
        retstep: 是否返回步长值（默认False）
        dtype: 指定数组的数据类型
        
        # 注意事项：
        1. 与arange()不同，默认包含结束值
        2. 实际步长为 (stop-start)/(num-1)（当endpoint=True时）

    举例：
    a = np.linspace(0, 10, num=5, endpoint=True)
    # a = [ 0.   2.5  5.   7.5 10. ]

    a, step = np.linspace(1, 2, num=3, retstep=True)
    # a = [1.  1.5 2. ], step = 0.5
"""
