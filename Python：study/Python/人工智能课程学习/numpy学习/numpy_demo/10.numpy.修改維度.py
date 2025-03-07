import numpy as np


def test01():
    arr = np.array([1, 2, 3, 4])
    arr1 = np.expand_dims(arr, axis=0)
    arr2 = np.expand_dims(arr, axis=1)

    print(arr1)
    print(arr2)


def test02():
    arr = np.array([[[1, 2, 3]]])
    print(arr.shape)
    # np.squeeze(): 数组降维，如果指定axis，则按照axis删除维度，前提：该维度值必须为1，否则抛异常
    # 如果不指定axis，则自动判断数组的维度是否为1，并且删除维度为1的维度
    arr1 = np.squeeze(arr, axis=0)
    print(arr1)

    arr2 = np.squeeze(arr, axis=1)
    print(arr2)

    # arr3 = np.squeeze(arr, axis=2)
    # print(arr3)

    arr4 = np.squeeze(arr)
    print(arr4)


def test03():
    arr = np.array([1, 2, 3])
    # newaxis:在数组的指定位置添加一个维度，可以在行或列添加维度
    arr1 = arr[np.newaxis, :]
    print(arr1)

    arr2 = arr[:, np.newaxis]
    print(arr2)


if __name__ == '__main__':
    test03()
