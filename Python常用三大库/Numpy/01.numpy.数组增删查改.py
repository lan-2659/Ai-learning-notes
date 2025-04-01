import numpy as np

# resize():修改数组形状
# 不受原数组个数限制，在修改新形状的时候，如果原数组元素个数不够，则重复遍历原数组元素进行填充
# 如果原数组元素个数大于新数组的元素个数，则丢弃
a = np.array([[1, 2, 3], [4, 5, 6]])
# 修改的形状中元素个数大于原数组的个数，则重复遍历原数组填充
b = np.resize(a, (3, 4))
print(b)
# 修改的新形状中元素个数小于原数组的元素个数，原数组中多余的元素丢弃
b = np.resize(a, (2, 2))
print(b)

# append():在数组尾部添加值(可以是数组)
a = np.array([[1, 2, 3], [4, 5, 6]])
# axis=None或未指定，返回一个一维数组
b = np.append(a, [1, 1, 1], axis=None)
print(b)
# axis=0，按行添加，values的列数要和原数组一致
# 要添加的values的维度要和原数组的维度相同才能添加
# b = np.append(a, [[1, 1, 1]], axis=0)
b = np.append(a, [[1, 1, 1], [1, 1, 1]], axis=0)
print(b)

# axis=1,按列添加，values的行数要和原数组的一致
b = np.append(a, [[1, 1, 1], [1, 1, 1]], axis=1)
print(b)

# insert():在原数组指定索引位置之前插入值
a = np.array([[1, 2, 3], [4, 5, 6]])
# 参数：
# arr:原数组
# obj：要插入的索引位置
# values：要插入的值
# axis：轴方向

# axis=None，返回一维数组，然后插入，最终返回一个一维数组
b = np.insert(a, 1, [6], axis=None)
print(b)
# axis=0,插入行，如果要插入的数组在行方向的维度为1，则自动进行广播，然后再插入
b = np.insert(a, 1, [6], axis=0)
print(b)
# axis=1,插入列，如果要插入的数组在列方向维度为1，则自动广播，再插入
b = np.insert(a, 1, [6], axis=1)
print(b)

# 错误示例
# 当axis=1时，要插入的values在行方向的元素个数要和原数组一致
# 当axis=0时，同理
# b = np.insert(a, 1, [6, 7, 8], axis=1)
# print(b)

# delete():删除指定索引位置的元素
# 参数：
# arr：要被删除的数组
# obj：删除元素的索引位置
# axis：轴方向

# 一维数组
a = np.array([1, 2, 3, 4, 5, 6])
b = np.delete(a, [2, 4])
print(b)

# 二维数组
a = np.arange(6).reshape(2, 3)
# axis=None,返回一维数组
b = np.delete(a, 1, axis=None)
print(b)
# axis=0,删除行
b = np.delete(a, 1, axis=0)
print(b)
# axis=1，删除列
b = np.delete(a, 1, axis=1)
print(b)

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
