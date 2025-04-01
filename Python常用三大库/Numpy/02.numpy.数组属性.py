import numpy as np

# shape:
# 1.返回一个形状元组
# 2.可以修改数组形状，直接修改原数组的形状
# 3.修改数组形状前提：数组的元素个数不能改变
a = np.array([1, 2, 3, 4, 5, 6])
print(a.shape)
# 形状赋值
a.shape = (2, 3)
print(a)
# ndim:返回数组的维度数，如：二维数组的ndim为2
print(a.ndim)

# flags：返回数组的内存信息
a = np.arange(6)
a.shape = (2, 3)
print(a.flags)
