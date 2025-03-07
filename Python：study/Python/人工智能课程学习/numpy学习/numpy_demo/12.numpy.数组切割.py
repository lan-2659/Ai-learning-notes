import numpy as np


def test01():
    arr = np.arange(1, 13).reshape(3, 4)
    print(arr)
    # hsplit():按照水平方向(按列)切割数组
    # 参数：
    # ary：要切割的数组
    # indices_or_sections:要切割的索引下标(不包含下标值)
    result = np.hsplit(arr, [1, 3])
    print(result[0])
    print(result[1])
    print(result[2])


def test02():
    arr = np.arange(1, 13).reshape(4, 3)
    print(arr)
    # vsplit():按照垂直方向(按行)切割数组
    # 参数：
    # ary：要切割的数组
    # indices_or_sections:要切割的索引下标(不包含下标值)
    result = np.vsplit(arr, [2])
    print(result[0])
    print(result[1])


if __name__ == '__main__':
    test02()
