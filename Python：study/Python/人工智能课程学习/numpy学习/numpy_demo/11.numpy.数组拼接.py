import numpy as np


def test01():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[7, 8], [9, 10]])
    # b = np.array([[7, 8], [9, 10], [11, 12]])
    print(a.shape, b.shape)
    # hstack():水平方向(按列)拼接数组，前提：数组的行数要一致
    c = np.hstack((a, b))
    print(c)


def test02():
    a = np.array([[1, 2], [3, 4], [5, 6]])
    b = np.array([[7, 8], [9, 10]])
    # vstack():垂直方向(按行)拼接数组，前提：数组的列数要一致
    c = np.vstack((a, b))

    print(c)


if __name__ == '__main__':
    test02()
