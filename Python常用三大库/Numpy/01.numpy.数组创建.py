import numpy as np

# array():创建数组
# object：可以输入一个列表，列表可以是一维或多维
# dtype：指定数组的数据类型，可选，如果不指定，则根据实际输入的数据类型自动判断
# order：可以指定内存的存储方式，C-按行，F-按列
# ndmin：设置数组的维度
a = np.array([1, 2, 3, 4], dtype=np.int32, order='F')
print(a)

a = np.array([1, 2, 3], ndmin=2)
print(a)
# shape：数组形状，返回一个元组，元组中的元素个数就是数组的维度数
# 如果数组是一个二维数组，形状元组的第一个元素表示行数，第二个元素表示列数
print(a.shape)

# zeros():根据指定的形状创建数组，元素默认以0填充
a = np.zeros((2, 3))
print(a)

# ones():根据指定的形状创建数组，元素默认以1填充
a = np.ones((4, 3))
print(a)

# full():根据指定的形状创建数组，元素可以使用指定值填充
a = np.full((3, 3), fill_value=5)
print(a)

# arange(start,stop,step):根据参数创建一个等差数列的数组(一维)
# start:起始值，stop：终止值(不包含),step:步长
a = np.arange(0, 10, 1)
print(a)
# 如果start为0，可以省略；如果step为1，可以省略
a = np.arange(10)
print(a)

# 生成0-9的偶数数组
a = np.arange(0, 10, 2)
print(a)
# 生成1-10之间奇数数组
a = np.arange(1, 11, 2)
print(a)

# 生成一个数组，元素降序排序
# 步长可以为负数，数组的元素则按降序排序，start值要比stop值大
a = np.arange(10, 0, -1)
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


