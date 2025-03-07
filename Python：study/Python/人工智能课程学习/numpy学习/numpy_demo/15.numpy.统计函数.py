import numpy as np


def test01():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    # amax():求数组的最大值
    # amin():求数组的最小值

    # axis=None，求整个数组中的最大值或最小值
    # axis=0，按列求最大值或最小值
    # axis=1，按行求最大值或最小值
    print(np.amax(a, axis=0))
    print(np.amin(a, axis=0))

    print(np.amax(a, axis=1))
    print(np.amin(a, axis=1))

    print(np.amax(a))
    print(np.amin(a))


def test02():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    # np.ptp(a): amax() - amin()，最大值 - 最小值

    # axis=None，求整个数组中的最大值减最小值
    # axis=0，按列求最大值减最小值
    # axis=1，按行求最大值减最小值
    print(np.ptp(a))

    print(np.ptp(a, axis=0))

    print(np.ptp(a, axis=1))


def test03():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    # np.median(a):求数组的中位数，如果数组元素个数为奇数，中位数是数组最中间的元素值，如果为偶数，中位数是数组最中间的两个元素值的平均值
    print(np.median(a))

    print(np.median(a, axis=0))

    print(np.median(a, axis=1))


def test04():
    a = np.array([[1, 2, 3], [4, 5, 6]])

    print(np.mean(a))
    print(np.mean(a, axis=0))
    print(np.mean(a, axis=1))


def test05():
    a = np.array([1, 2, 3, 4, 5])
    w = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
    # np.average(a, weights=w):计算数组的加权平均值：所有元素值乘以对应的权重相加，除以所有权重之和
    print(np.average(a, weights=w))


def test06():
    a = np.array([1, 2, 3, 4, 5])
    # np.var():计算数组元素的方差
    # 参数：
    # ddof：默认使用总体方差，如果设置为1，则表示使用样本方差
    print(np.var(a, ddof=1))

    # np.std()：计算标准方差
    print(np.std(a, ddof=1))


if __name__ == '__main__':
    test06()
