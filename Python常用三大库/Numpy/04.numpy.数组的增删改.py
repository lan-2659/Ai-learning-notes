# 数组的添加
"""
# 在数组尾部置添加值

    numpy.append()方法：

        语法格式：
        numpy.append(arr, values, axis=None)
        参数：
        arr：要添加值的数组
        values：向arr数组中添加的值，添加位置是末尾，values的形状需兼容(在注意事项中有详细说明)
        axis：轴方向(即在哪个维度进行添加)

        注意事项：
        返回添加后的数组
        如果axis=None，则返回一维数组（且此时values的形状无要求）
        values的传入形状要求：
            原数组形状：(A, B, C, ..., N)
            插入轴：
            axis=0：values 形状需为 (K, B, C, ..., N),K>=1
            axis=1：values 形状需为 (A, K, C, ..., N),K>=1
            axis=2：values 形状需为 (A, B, K, ..., N),K>=1
            ...
            axis=N：values 形状需为 (A, B, C, ..., K),K>=1

        举例：
        import numpy as np

        a = np.array([[1, 2, 3], [4, 5, 6]])    # a 的形状是 (2, 3)

        b = np.append(a, [1, 1, 1], axis=None)  # axis=None，返回一维数组
        print(b)

        c = np.append(a, [[7, 8, 9]], axis=0)   # axis=0在第一个维度添加，添加的数组形状为 (1, 3)（总维度数与原数组相同，元素个数只有第一个维度不同）
        print(c)

        d = np.append(a, [[7], [8]], axis=1)    # axis=1在第二个维度添加，添加的数组形状为 (2, 1)（总维度数与原数组相同，元素个数只有第二个维度不同）
        print(d)
        

# 在数组的指定位置添加值

    numpy.insert()方法：

        语法格式：
        numpy.insert(arr, obj, values, axis=None)
        参数：
        arr：要插入值的数组
        obj：要插入值的索引位置 (原位置元素往后移动)，可以是整数、切片、列表或整数数组
        values：要插入的值，values的形状需兼容(在注意事项中有详细说明)
        axis：轴方向(需要添加元素的维度)，默认为 None

        注意事项：
        返回添加后的数组
        如果axis=None，则返回一维数组（且此时values的形状无要求）
        values的传入形状要求：
            原数组形状：(A, B, C, ..., N)
            插入轴：
            axis=0：values 形状需为 (K, B, C, ..., N),K>=1 或可广播为这个形状的其它形状
            axis=1：values 形状需为 (A, K, C, ..., N),K>=1 或可广播为这个形状的其它形状
            axis=2：values 形状需为 (A, B, K, ..., N),K>=1 或可广播为这个形状的其它形状
            ...
            axis=N：values 形状需为 (A, B, C, ..., K),K>=1 或可广播为这个形状的其它形状

        举例：
        import numpy as np
        a = np.array([[1, 2, 3], [4, 5, 6]])     # a 的形状是 (2, 3)

        b = np.insert(a, 1, [6], axis=None)      # axis=None，返回一维数组
        print(b)
        c = np.insert(a, 1, [[6], [7]], axis=1)  # axis=1在第二个维度插入值，values形状为 (2, 1)，满足条件
        print(c)
        d = np.insert(a, 1, [[6], [7]], axis=0)  # axis=0在第一个维度插入值，values形状为 (2, 1)，不满足条件但可以广播为 (2, 3)
        print(d)
        e = np.insert(a, [1, 2], [6, 7], axis=1) # axis=1在第二个维度的第1、2位置分别插入值6、7，每个对应位置的values形状都是(1,)，不满足条件但可以广播为 (2, 1)
        print(e)
"""


# 数组的删除
"""
# 删除指定位置的元素

    numpy.delete()方法：

        语法格式：
        numpy.delete(arr, obj, axis=None)
        参数：
        arr: 需要删除元素的数组
        obj: 要删除的元素的索引（可以是整数、切片、列表或整数数组）。
        axis: 沿哪个轴删除元素(即待删除元素所在的维度)。默认为 None

        注意事项：
        返回删除元素后的数组
        若axis=None，则先将数组展平为一维，再删除元素（且会返回一个一维数组）

        举例:
        import numpy as np
        arr = np.arange(24).reshape(2, 3, 4)    # 形状 (2,3,4)
        
        result = np.delete(arr, 1, axis=2)      # 删除第三个维度（axis=2）的索引1
        print(result)   
        
        result = np.delete(arr, 2)      # 展平后删除索引2的元素（原数组中第3个元素）
        print(result)                   # 返回的是一个一维数组


# 删除数组中的重复元素(数组去重)

    numpy.unique()方法：

        语法格式：
        numpy.unique(arr, return_index=False, return_inverse=False, return_counts=False, axis=None)：
        参数：
        arr：要去重的数组
        return_index：如果为True，返回新数组元素在原数组中的位置(索引)
        return_inverse：如果为True，返回原数组元素在新数组的位置(逆索引)
        return_counts：如果为True，返回新数组元素在原数组中出现的次数
        axis：轴方向（即需要在数组的那一个维度进行去重），默认为 None

        注意事项：
        默认返回排序后(升序)的唯一值数组
        当 axis=None 时，原数组会被展开成一个一维数组进行去重，然后返回一个一维数组
        return_index、return_inverse、return_counts这三个参数用于返回附加信息(附加信息会和新数组打包成一个元组返回，元组第一个元素是新数组)

        举例：
        import numpy as np

        a = np.arange(1, 65).reshape(4, 4, 4)

        b = np.unique(a)    # axis=None，返回一维数组
        print(b)
"""

 
# 数组的修改
"""
# 修改数组的形状

    numpy.resize()方法：

        语法格式：
        numpy.resize(arr, new_shape)
        参数：
        arr：要修改的数组
        new_shape：修改后的数组形状

        注意事项：
        返回修改后的数组
        修改后的数组的形状可以任意指定(不受原数组个数限制，在修改新形状的时候，如果原数组元素个数不够，则重复遍历原数组元素进行填充)
        如果原数组元素个数大于新数组的元素个数，则丢弃多余的元素

        举例：
        import numpy as np
        a = np.array([[1, 2, 3], [4, 5, 6]])
        
        b = np.resize(a, (3, 4))        # 修改的形状中元素个数大于原数组的个数，则重复遍历原数组填充
        print(b)
        
        b = np.resize(a, (2, 2))        # 修改的新形状中元素个数小于原数组的元素个数，原数组中多余的元素丢弃
        print(b)
"""

import numpy as np

# argwhere():默认返回非0元素对应的索引坐标[[行下标,列下标],[行下标,列下标]...]
a = np.arange(6).reshape(2, 3)
b = np.argwhere(a)
print(b)
# 也可以使用布尔索引获取满足条件的索引坐标
b = np.argwhere(a > 3)
print(b)

# argmax():获取列表中最大值对应的下标索引
a = np.array([1, 3, 2, 5, 3])
b = np.argmax(a)
print(b)

# where():默认返回非0元素对应的索引下标，返回的结果是元组，元组中元素是行索引下标数组和列索引下标数组
# 可以结合整数数组索引来获取指定位置的元素
# 和argwhere一样，也可以使用布尔索引获取满足条件的索引下标
a = np.arange(6).reshape(2, 3)
b = np.where(a)
print(b)

# 布尔索引
b = np.where(a > 3)

c = a[b[0], b[1]]
print(c)

# unique():去重
# 参数：
# return_index:如果为True，返回新数组元素在原数组中的位置(索引)
# return_inverse:如果为True，返回原数组元素在新数组的位置(逆索引)
# return_counts:如果为True，返回新数组元素在原数组中出现的次数
# axis：轴方向

a = np.array([1, 2, 2, 3, 4, 4, 5])
b = np.unique(a)
print(b)

b, idx = np.unique(a, return_index=True)
print(b, idx)

b, inv_idx = np.unique(a, return_inverse=True)
print(b, inv_idx)

b, count = np.unique(a, return_counts=True)
print(b, count)

# 二维数组去重
a = np.array([[1, 2], [2, 3], [1, 2]])
b = np.unique(a, axis=0)
print(b)
