import numpy as np

# 创建两个形状不同的数组，行数一致
# arr1 = np.array([[1, 2], [3, 4]])
# arr2 = np.array([[5], [6]])
# print(arr1.shape) 	# (2, 2)
# print(arr2.shape)	# (2, 1)
#
# # 使用 hstack 水平堆叠数组
# result = np.hstack([arr1, arr2])
# print(result)

arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[1], [2], [3]])
print(arr1 + arr2)