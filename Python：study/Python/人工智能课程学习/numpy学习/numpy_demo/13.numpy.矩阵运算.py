import numpy as np


def test01():
    a = np.arange(1, 5)
    b = np.arange(5, 9)
    # np.dot():当数组是一维数组时，该方法为点积运算；当数组为二维数组时，该方法为矩阵乘法
    print(np.dot(a, b))


def test02():
    a = np.arange(1, 5).reshape(2, 2)
    b = np.arange(5, 9).reshape(2, 2)

    print(np.dot(a, b))


def test03():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])

    # np.matmul():矩阵乘法运算
    result = np.matmul(a, b)
    print(result)

def test04():
    a = np.array([[1, 2], [3, 4]])
    # np.linalg.det():行列式计算，前提：必须时方阵
    det = np.linalg.det(a)
    print(det)

if __name__ == '__main__':
    test04()
