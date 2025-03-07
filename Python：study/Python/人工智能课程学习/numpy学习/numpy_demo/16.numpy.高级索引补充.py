import numpy as np


def test01():
    data = np.arange(1, 13).reshape(3, 4)
    # 行级别的数组整数索引列表
    # 通过广播机制将行索引数组和列索引转换相同的形状，然后再按照数组整数索引获取对应的元素
    print(data[[[2], [1]], [0, 1, 2]])


if __name__ == '__main__':
    test01()
