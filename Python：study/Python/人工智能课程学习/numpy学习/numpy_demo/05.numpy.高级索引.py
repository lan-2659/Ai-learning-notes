import numpy as np


def test01():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(arr)

    # 整数索引：通过行标和列标获取数组中任意位置的元素
    # 通过整数索引获取元素1、5、8
    # 获取到新数组是原数组的一个副本，修改新数组中的元素值不影响原数组
    arr1 = arr[[0, 1, 2], [0, 1, 1]]
    print(arr1)

    arr1[0] = 100

    print(arr1)
    print(arr)


def np_reshape():
    arr = np.array([[1, 2, 3], [4, 5, 6]])

    arr1 = arr.reshape(3, 2)

    arr1[0][0] = 100

    print(arr1)

    print(arr)


# 一维数组布尔索引
def test02():
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    idx = arr > 5
    print(idx)

    print(arr[idx])

    arr1 = arr[arr > 5]
    print(arr1)

    arr1[0] = 100
    print(arr1)
    print(arr)


def test03():
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    # 与或非运算符：与-&，或-|，非-~
    arr1 = arr[(arr > 5) | (arr < 9)]
    print(arr1)


if __name__ == '__main__':
    test02()
