import numpy as np

'''
广播：把两个不同形状的数组变成相同形状的数组，然后进行算数运算
要求：数组的维度为1才能广播
'''


def test01():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([10, 20, 30])

    c = a + b
    print(c)


def test02():
    a = np.array([[1], [2]])
    b = np.array([10, 20, 30])

    c = a + b

    print(c)


def test03():
    a = np.array([1, 2, 3])
    # 标量1被广播为[1 1 1]
    b = a + 1

    print(b)


def test04():
    a = np.array([1, 2, 3, 4])
    b = np.array([1, 2, 3, 4]).reshape(4, 1)
    # a的形状为(4,)，行被广播，形状为(4,4)
    # b的形状为(4,1)，列被广播，形状(4,4)
    # a和b形状相同后，进行对应元素的相加运算
    c = np.add(a, b)

    print(c)


'''
广播错误案例
'''


def test05():
    # a的形狀(3,3)
    # b的形狀(2,3)
    # a+b不能运算，原因是b数组的行维度不为1
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([[1, 2, 3], [4, 5, 6]])

    c = a + b

    print(c)


if __name__ == '__main__':
    test05()
