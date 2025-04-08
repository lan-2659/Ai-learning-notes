# ndarray对象的运算
"""
# 语法格式：
ndarray1 + ndarray2
ndarray1 - ndarray2
ndarray1 * ndarray2
ndarray1 / ndarray2
ndarray1 ** ndarray2
ndarray1 % ndarray2
ndarray1 // ndarray2
ndarray1 & ndarray2
ndarray1 | ndarray2

# 注意事项：
1、ndarray对象的运算本质是对应位置的元素分别进行运算，将所有运算结果组成一个新数组返回
2、两个ndarray对象进行运算时，两个数组的形状必须相同，或者其中一个可以广播也行
3、ndarray对象与某一值进行运算时，其实是将这个单个值运ndarray对象的每一个元素分别进行了运算，将所有运算结果组成一个新数组返回
"""


# ndarray对象的索引
"""
# 整数数组索引：

    # 语法格式：
    ndarray[index1, index2, ..., indexN]
        index1, index2, ..., indexN 可以是整数、整数数组
        N 的最大值等于数组的维度数，且每个索引对应一个维度

    # 注意事项：
    1. 返回数据的副本（与原数组内存独立）
    2. 通过整数或整数数组明确指定索引位置

    # 举例：
    import numpy as np

    a = np.arange(12).reshape(4, 3)

    start = a[0, 0]                               # 访问单个位置 (0,0) 
    print(start)
    corners = a[[0, 0, -1, -1], [0, -1, 0, -1]]   # 访问多个位置 (0,0), (0,-1), (-1,0), (-1,-1)
    print(corners)  

    rows = np.array([[0], [2]])     # 形状 (2,1)
    cols = np.array([[1, 0]])       # 形状 (1,2)
    print(a[rows, cols])            # 输出 [[2 1],[8 7]]

    copy_data = a[[0, 2]]
    copy_data[0, 0] = 100           # 修改副本不会影响原数组
    print(a[0,0] == 0)              # 输出 True

    
# 布尔索引：

    # 语法格式：
    ndarray[index1, index2, ..., indexN]
        index1, index2, ..., indexN 为布尔数组
        N 的最大值等于数组的维度数，且每个索引对应一个维度

    # 注意事项：
    1. 返回数据的副本（与原数组内存独立）
    2. 通过布尔数组（掩码）筛选元素（True保留，False过滤）
    3. 布尔数组需与该维度的数组形状兼容（可广播）
    4. 支持组合逻辑运算（&, |, ~）

    # 举例：    
    import numpy as np

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    mask = a > 5                   # [False False False False False  True  True  True  True]
    print("筛选结果:", a[mask])     # [6 7 8 9]

    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("行条件筛选列:\n", a[:, a[1] > 3])        # 只在某一维度进行布尔索引时，需注意布尔数组格式

    a = np.arange(1, 11)
    mask = (a % 2 == 0) & (a > 5)      # 多条件创建布尔数组时，必须用括号包裹子条件
    print("偶且大于5:", a[mask])


# 切片：

    # 语法格式：
    ndarray[index1, index2, ..., indexN]
        index1, index2, ..., indexN 可以是 切片或省略号 (...)
        N 的最大值等于数组的维度数，且每个索引对应一个维度

    注意事项：
    ndarray对象切片返回的是一个ndarray对象（原对象的视图）
    ndarray对象切片时可以在一个 [] 中进行连续切片，每个维度的切片之间用','分隔
    每个维度的切片的start和stop可以越界(两个同时越界都可以)
    切片时会先根据步长的正负判断区间[start:end)是否可以进行正向或反向遍历，只要有一个维度的切片不行就会直接返回一个空一维数组
    切片中'...'符号的具体作用在举例中有说明

    举例：
    import numpy as np

    arr = np.arange(64).reshape(4, 4, 4)

    # ndarray数组 的连续切片只能从第一个维度开始，且一般不能跳维度(除非使用 ... 语法)
    print(arr[0:2])             # 只对第一个维度切片
    print(arr[0:2, 0:2])        # 只对前两个维度切片
    print(arr[0:2, 0:2, 0:2])   # 对所有维度切片

    # 下面的方法在高维度数组中具有泛用性
    print(arr[0:2, ...])        # 对第一个维度切片，... 表示对一维后面的所有维度全切
    print(arr[..., 0:2])        # 对最后一个维度切片，... 表示对最后一维前面的所有维度全切
    print(arr[0:2, ..., 0:2])   # 对第一个和最后一个维度切片，... 表示中间的的所有维度全切
"""
