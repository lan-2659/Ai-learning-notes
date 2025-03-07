import numpy as np


def test01():
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    # 只遍历数组中的第一维元素
    for i in arr:
        print(i)


def test02():
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    # np.nditer():遍历数组的迭代器
    # 参数：
    # order：按照存储方式遍历，C-按行遍历，F-按列遍历，默认为C
    # flags:
    #  - multi_index:除了返回元素值外，还返回元素值对应的下标索引
    #  - external_loop：按行遍历元素，遍历完所有行之后封装为数组输出，如果order='F'，输出时按列输出多个数组(默认按行遍历，遍历完所有行封装数组)
    # op_flags:设置数组元素的读写操作
    #  - readonly：只读
    #  - readwrite: 可读写
    #  - writeonly: 只写
    iter = np.nditer(arr, order='F')

    for i in iter:
        print(i, end=' ')


def test03():
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    iter = np.nditer(arr, flags=['multi_index'])
    for i in iter:
        print(i, iter.multi_index)

    iter1 = np.nditer(arr, flags=['external_loop'])

    for i in iter1:
        print(i)

    iter2 = np.nditer(arr, flags=['external_loop'], order='F')
    for i in iter2:
        print(i)


def test04():
    arr = np.array([[1, 2, 3], [4, 5, 6]])

    # iter = np.nditer(arr, op_flags=['readonly'])
    iter = np.nditer(arr, op_flags=['readwrite'])
    for x in iter:
        print(type(x))
        x[...] = x * 2
        print(x)


if __name__ == '__main__':
    test01()
