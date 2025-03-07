import numpy as np


def test01():
    arr = np.full((2, 2), 5)
    # flat属性：将多维数组降维成一维数组，返回的是一个迭代器，用来遍历数组
    for i in arr.flat:
        print(i, end=' ')


def test02():
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    # flatten()：将多维数组降维成一维数组，返回的新数组是原数组的一个副本，修改新数组元素不影响原数组
    arr1 = arr.flatten(order='F')
    print(arr1)

    arr1[0] = 100

    print(arr)


def test03():
    arr = np.array([[1, 2, 3], [4, 5, 6]])

    # ravel():数组降维成一维数组，返回的新数组是原数组的一个视图，修改新数组元素会影响原数组
    arr1 = arr.ravel()

    print(arr1)

    arr1[0] = 100

    print(arr)

if __name__ == '__main__':
    test03()
