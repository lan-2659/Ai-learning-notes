import numpy as np


# 给定一个二维数组 arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])，提取出主对角线上的元素，并将这些元素替换为它们的平方。
def test01():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    size = arr.shape[0]
    print(size)

    index = np.arange(size)
    print(index)

    b = arr[index, index]
    print(b)

    c = b ** 2
    print(c)

    arr[index, index] = arr[index, index] ** 2
    print(arr)

    # diag():获取主对角线的元素
    d = np.diag(arr)
    print(d)

    # diag_indices_from(): 获取主对角线元素对应的下标
    # 数据格式：(array([0, 1, 2]), array([0, 1, 2]))，是一个元组，第一个元素是行下标数组，第二个元素是列下标数组
    idx = np.diag_indices_from(arr)
    print(idx)


# 给定一个二维数组 arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])，将其按行分割为两个子数组：第一个子数组包含前两行，第二个子数组包含最后一行。
def test02():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    res = np.vsplit(arr, [2])

    print(res[0], res[1])

    # 切片
    arr1 = arr[:2, ...]
    arr2 = arr[2:, ...]

    print(arr1, arr2)


# 给定一个二维数组 arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])，找出所有大于5的元素的索引，并将这些元素替换为它们的平方根。
def test03():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)

    index = np.argwhere(arr > 5)
    print(index)

    res = np.hsplit(index, [1])
    rows = res[0]
    cols = res[1]

    b = arr[rows, cols]
    print(b)

    c = np.sqrt(b)
    print(c)

    # 原地修改
    # arr[rows,cols] = np.sqrt(arr[rows,cols])
    # print(arr)

    # where(): 获取元素的索引下标
    # 结果数据格式：(array([1, 2, 2, 2], dtype=int64), array([2, 0, 1, 2], dtype=int64))，是一个元组，第一个数组是行下标，第二个数组是列下标
    # idx = np.where(arr > 5)
    # print(idx)


# 给定一个 NumPy 数组，编写一个函数，使用广播机制将数组的每个元素加上一个给定的标量值，并返回结果数组。
def test04():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    b = arr + 1

    print(b)


# 创建一个二维 NumPy 数组，其中包含从 1 到 100 的整数，然后使用高级索引提取所有偶数，并计算这些偶数的总和。
def test05():
    a = np.arange(1, 101).reshape(4, -1)
    print(a)

    b = a[a % 2 == 0]
    print(b)

    # np.sum(b):计算数组中元素总和
    sum = np.sum(b)
    print(sum)


# 编写一个函数，返回数组沿指定轴的方差。
def test06():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    v = np.var(arr, axis=1)
    print(v)


# 实现一个函数，它接受一个 NumPy 数组，并返回该数组中所有唯一元素的列表。
def test07():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    b = np.unique(arr)
    print(b)


# 使用Python的NumPy库，创建一个形状为(4,4)的二维数组，并且初始化所有元素为-1。
def test08():
    arr = np.full((4, 4), -1)
    print(arr)

    arr = np.zeros((4, 4)) - 1
    print(arr)

    arr = np.ones((4, 4)) * (-1)
    print(arr)


# 假设数组：[1, 2, 3, 4, 3, 5, 3]，删除数组中所有值为3的元素。
def test09():
    arr = np.array([1, 2, 3, 4, 3, 5, 3])
    # 使用布尔索引过滤
    b = arr[arr != 3]
    print(b)

    # 使用where和delete删除
    idx = np.where(arr == 3)
    print(idx)
    c = np.delete(arr, idx)
    print(c)


if __name__ == '__main__':
    test09()
