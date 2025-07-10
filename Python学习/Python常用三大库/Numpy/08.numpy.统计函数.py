import numpy as np

# amin()和amax():查找最小值和最大值
# 参数：axis=0，按垂直方向查找，axis=1，按水平方向查找，axis=None，查找整个数组中(可以理解在一维数组中查找)的最小值和最大值
a = np.array([[1, 23, 4, 5, 6], [1, 2, 333, 4, 5]])
print(np.amin(a, axis=1))
print(np.amax(a, axis=1))

print(np.amin(a, axis=0))
print(np.amax(a, axis=0))

print(np.amin(a))
print(np.amax(a))

# ptp():最大值-最小值
# 参数：axis=0，按垂直方向查找，axis=1，按水平方向查找，axis=None，查找整个数组中(可以理解在一维数组中查找)的最小值和最大值然后相减
a = np.array([[1, 23, 4, 5, 6], [1, 2, 333, 4, 5]])
print(np.ptp(a, axis=1))
print(np.ptp(a, axis=0))
print(np.ptp(a))

# median()：中位数
# 数组中元素按照从小到大排列后，取数组中间的值，如果元素个数为奇数，则直接取中间的元素值，如果为偶数，则取中间两个数的平均值
# axis=1，按水平方向计算，axis=0.按垂直方向计算，axis=None，转换一维数组然后计算
a = np.array([[1, 2, 3], [4, 5, 6]])
print(np.median(a, axis=0))
print(np.median(a, axis=1))
print(np.median(a))

# mean():算术平均值
# axis=1，按水平方向计算，axis=0：按垂直方向计算，axis=None，转换一维数组然后计算
a = np.array([[1, 2, 3], [4, 5, 6]])
print(np.mean(a, axis=1))
print(np.mean(a, axis=0))
print(np.mean(a))

# average():加权平均值
# 数组中所有元素乘以对应的权重之和，除以所有权重之和
# 如果所有权重之和为1，则表示为概率中的期望值(均值)

a = np.array([1, 2, 3, 4, 5])
weights = [1, 2, 3, 2, 2]
print(np.average(a, weights=weights))

# var():求方差
# numpy中默认是总体方差，如果ddof=1，表示是样本方差
a = np.array([1, 2, 3, 4, 5])
print(np.var(a))
print(np.var(a, ddof=1))

# std():标准差：方差开根号
a = np.array([1, 2, 3, 4, 5])
print(np.std(a))
print(np.std(a, ddof=1))
# 求和
print(np.sum(a))
# 求平方根
print(np.sqrt(np.sum(a)))
