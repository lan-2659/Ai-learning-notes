import pandas as pd
import numpy as np


def test01():
    # 创建一个空Series对象
    # s = pd.Series()
    s = pd.Series(dtype=np.int32)
    print(s)

    # 通过python列表创建Series，如果不指定index，则默认从0开始
    s = pd.Series([1, 2, 3, 4, 5, 6])
    print(s)

    # 指定index标签
    index = ['a', 'b', 'c', 'd', 'e']
    s = pd.Series([3, 4, 5, 6, 7], index=index)
    print(s)

    # 通过ndarray创建Series对象
    arr = np.array([1, 2, 3, 4, 5])
    s = pd.Series(arr)
    print(s)

    # 通过字典创建Series，字典中的key是value的标签
    dic = {'id': 1, 'name': 'zhangsan', 'age': 20}
    s = pd.Series(dic)
    print(s)

    # 通过标量创建Series
    s = pd.Series(5)
    print(s)

    # 通过标量创建Series，同时指定index，如果index个数多于标量，则标量按照index的个数进行填充
    s = pd.Series(5, index=[1, 2, 3, 4, 5])
    print(s)


def test02():
    s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
    print(s)
    # 根据标签名获取元素
    b = s['b']
    print(b)
    # 根据数组下标获取元素
    b1 = s[1]
    print(b1)

    # 根据标签名修改元素值
    s['a'] = 100
    print(s)
    # 根据下标修改元素值
    s[1] = 200
    print(s)


def test03():
    s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
    # 以数组方式返回Series对象的行标签列表
    print(s.axes)
    # 返回Series对象的数据类型
    print(s.dtype)
    # 判断Series对象是否为空
    print(s.empty)

    # 返回Series对象的维度数
    print(s.ndim)
    # 返回Series对象的元素总数
    print(s.size)
    # 返回ndarray类型的元素数组
    print(s.values)
    # 返回Series对象的行标签列表
    print(s.index)


def test04():
    s = pd.Series([1, 2, 3, 4, 5, 6])
    # head():获取前n行数据，默认n=5
    print(s.head(3))
    # tail():获取后n行数据，默认n=5
    print(s.tail(3))

    s1 = pd.Series([1, 2, 3, 4, None])
    # isnull():判断Series对象中的原始是否为空，如果为空返回True，否则返回False
    print(s1.isnull())
    # notnull()：判断Series对象中的原始是否为空，如果不为空返回True，否则返回False
    print(s1.notnull())


if __name__ == '__main__':
    test04()
