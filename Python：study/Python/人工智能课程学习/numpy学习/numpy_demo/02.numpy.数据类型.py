import numpy as np


def test01():
    # arr = np.array([[1,2,3],[4,5,6]],dtype=np.int8)
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float16)
    print(arr)
    print(type(arr))
    print(arr.dtype)


def test02():
    # 数据类型标识符：i、f等，在i或f后添加数据类型长度，如：1表示数据类型占用的字节数，i1-->np.int8,i2-->np.int16,i4-->int32，f同i
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype='i4')
    print(arr.dtype)


def test03():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    # itemsize:数组每个元素占用的字节数
    print(arr.itemsize)


def test04():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64, order='F')
    # flags:查看數組的内存信息
    print(arr.flags)

    # 一维数组的内存信息：行优先和列优先都为True
    arr1 = np.array([1, 2, 3, 4], order='F')
    print(arr1.flags)


if __name__ == '__main__':
    test04()
