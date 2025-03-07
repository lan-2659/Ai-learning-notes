import numpy as np


def test01():
    arr = np.arange(1, 13).reshape(3, 4)
    print(arr)
    # transpose():数组转置，返回的是原数组的视图
    # T属性：数组转置，返回的是原数组的视图
    arr1 = arr.transpose()
    print(arr1)

    arr1[0][0] = 100

    # print(arr)

    arr2 = arr.T

    arr2[0][1] = 200
    print(arr2)
    print(arr)


if __name__ == '__main__':
    test01()
