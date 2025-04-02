# NumPy 定义了一个 n 维数组对象，简称 ndarray 对象


# ndarray对象 是一个由一系列相同类型元素组成的数组集合
# 数组中的每个元素都占有大小相同的内存块，且这些内存块是连续的


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

    numpy.array(object, dtype=None, copy=True, order='C', ndmin=0)  

        这个方法会返回一个 ndarray对象，返回对象的逻辑结构由object参数决定
        # 参数说明：
        object: 可以是任何可转换为数组的对象(如列表、元组、其它ndarray对象等)(并非是可迭代对象，如字符串、集合、字典等，这些会创建失败但不报错)
        dtype: 指定数组的数据类型，可选，如果不指定，则根据实际输入的数据类型自动判断
        copy: 默认为True，表示新数组中的数据与原数据是否完全独立（如果为False，则新数组与原数组共享同一批数据；如果为True，则会创建原数据的副本作为新数组的数据）
        order: 可以指定元素在内存中的存储方式(但不会影响逻辑结构)，C-按行，F-按列，默认为 C
        ndmin: 指定最小维度的参数，确保生成的数组至少具有指定的维度
        
        # 特别注意：copy在绝大多数情况下都是True（不管有没有指定copy为False），只有当object参数为ndarray对象，且dtype和order都不指定（或指定为和object相同的类型）时，copy=False才生效

    举例：
    arr = np.array([1, 2], dtype=int, copy=True, order='F', ndmin=0) 


# zeros() 方法：

    # :根据指定的形状创建数组，元素默认以0填充
    a = np.zeros((2, 3))
    print(a)

# ones() 方法

    # 根据指定的形状创建数组，元素默认以1填充
    a = np.ones((4, 3))
    print(a)

# full() 方法:

    # 根据指定的形状创建数组，元素可以使用指定值填充
    a = np.full((3, 3), fill_value=5)
    print(a)

# arange() 方法:
    # arange(start,stop,step):根据参数创建一个等差数列的数组(一维)
    # start:起始值，stop：终止值(不包含),step:步长
    a = np.arange(0, 10, 1)
    print(a)
    # 如果start为0，可以省略；如果step为1，可以省略
    a = np.arange(10)
    print(a)



# linspace():生成一个等差数列的一维数组，按份数生成
# start：起始值
# stop：终止值(默认包含)
# num: 要分成等差数列的份数，默认为50
# endpoint：默认为True，表示包含stop，如果为False表示不包含stop
a = np.linspace(1, 10, 20)
print(a)
# step=(stop-start)/(num-1)
a, step = np.linspace(1, 10, 20, retstep=True)
print(a, step)

a = np.array([1,2,3,4],dtype=np.int32)
print(a)
"""
