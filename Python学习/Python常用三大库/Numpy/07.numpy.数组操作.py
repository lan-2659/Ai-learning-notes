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


import numpy as np

# reshape():修改数组的形状
# 修改后返回一个新数组，不直接修改原数组
# 返回的新数组是原数组的视图，修改新数组会影响原数组
# 前提：新数组和原数组的元素个数要保持一致

a = np.array([1, 2, 3, 4, 5, 6])
a1 = a.reshape((2, 3))
print(a1)
print(a)
# 返回的新数组是原数组的视图，修改新数组会影响原数组
a1[0, 0] = 100
print(a)
# -1:占位符，numpy会自动计算该占位符的维度
a2 = a.reshape((2, -1))
print(a2)

a = np.arange(1, 10).reshape(3, 3)
print(a)

# flat属性：返回一个一维数组迭代器，可以使用循环来遍历数组中的元素
a = np.arange(1, 10).reshape(3, 3)
for i in a.flat:
    print(i, end=' ')
print()

# flatten():返回一个一维数组，是原数组的副本，修改新数组不影响原数组
a = np.arange(1, 10).reshape(3, 3)
a1 = a.flatten()
print(a1)
a1[0] = 100
print(a1)
print(a)

# ravel():返回一个一维数组，是原数组的视图，修改新数组会影响原数组
a = np.arange(1, 10).reshape(3, 3)
a1 = a.ravel()
print(a1)
a1[0] = 100
print(a)

# 数组转置
# T属性
# transpose()

a = np.arange(1, 7).reshape(2, 3)
print(a.T)
print(np.transpose(a))

# expand_dims(arr,axis):根据指定的轴方向进行升维
# axis=0，按行升维，axis=1.按列升维
a = np.array([1, 2, 3])
a1 = np.expand_dims(a, axis=0)
print(a1)
a2 = np.expand_dims(a, axis=1)
print(a2)

# 升维结合广播
a = np.array([1, 2, 3])
b = 2
# 对a按行升维
c = np.expand_dims(a, axis=0)

d = c + b
print(d)

# squeeze(arr,axis):根据指定的轴进行降维
# 降维前提：所在轴上的维度数必须为1，才能做降维操作
# axis：按照轴方向进行降维，0-按最外层降维，1-第二层降维，以此类推
# 假设数组为二维数组：
# axis=0，按行降维
# axis=1，按列降维
# axis=None或不指定，则对所有维度数为1的项降维

a = np.array([[[1, 2, 3]]])
print(a.shape)

# 按最外层降维
a1 = np.squeeze(a, axis=0)
print(a1)
# 按第二层降维
a2 = np.squeeze(a, axis=1)
print(a2)
# 按第三层降维
# 该层的维度数不为1，抛异常：ValueError: cannot select an axis to squeeze out which has size not equal to one
# a3 = np.squeeze(a, axis=2)
# print(a3)
# axis=None或不指定，则删除所有维度数为1的项
a4 = np.squeeze(a)
print(a4)

# 数组拼接
# hstack(tuple):参数是一个元组，按水平方向拼接(按列),要求：行数一致
# vstack(tuple):参数是一个元组，按垂直方向拼接(按行),要求：列数一致

# 按列拼接
a = np.array([[1, 2], [3, 4]])
b = np.array([[5], [6]])
c = np.hstack((a, b))
print(c)

# 按行拼接
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[7, 8, 9], [10, 11, 12]])
c = np.vstack((a, b))
print(c)

# hsplit(arr,[index]):按照水平方向(按列)切割数组，index表示数组的列索引位置
# vsplit(arr,[index]):按照垂直方向(按行)切割数组，index表示数组的行索引位置
a = np.arange(1, 13).reshape(3, 4)
# 水平方向切割
result = np.hsplit(a, [1, 3])
# print(result)
print(result[0])
print(result[1])
print(result[2])

result = np.vsplit(a, [1])
print(result[0])
print(result[1])

# dot：矩阵运算
# 对于一维数组，dot做点积运算
# 对于二维数组，dot做矩阵运算

# 点积
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.dot(a, b)
print(c)

# 矩阵运算
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[1, 2], [3, 4], [5, 6]])
c = np.dot(a, b)
print(c)

# matmul():专门做矩阵运算
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[1, 2], [3, 4], [5, 6]])
c = np.matmul(a, b)
print(c)

# det():计算方阵的行列式(必须是方阵才能计算行列式),返回结果是一个标量
a = np.array([[1,2],[3,4]])
b = np.linalg.det(a)
print(b)

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.linalg.det(a)
print(b)
