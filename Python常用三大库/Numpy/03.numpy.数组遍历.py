import numpy as np

# 二维数组直接使用for循环遍历
# 输出结果是数组中的每个一维数组
a = np.array([[1, 2, 3], [4, 5, 6]])
for i in a:
    print(i)

# nditer:遍历迭代器，可以用来遍历多维数组
# 控制参数：
# 1.order：设置遍历数组的顺序，C-按行遍历(默认)，F-按列遍历
a = np.array([[1, 2, 3], [4, 5, 6]])
for i in np.nditer(a, order='C'):
    print(i, end=' ')
print()

for i in np.nditer(a, order='F'):
    print(i, end=' ')
print()

# 2.flags:指定迭代器的额外行为
# 参数值：
# multi_index：返回元素对应的下标索引
# external_loop：将遍历的单个元素添加到一个一维数组，遍历完成后输出一维数组

a = np.array([[1, 2, 3], [4, 5, 6]])
it = np.nditer(a, flags=['multi_index'])
for i in it:
    print(i, it.multi_index)

for i in np.nditer(a, flags=['external_loop']):
    print(i)
