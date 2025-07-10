import pandas as pd


def test01():
    data = {
        'A': [1, 2, 3, 4],
        'B': [5, 6, 7, 8]
    }

    df = pd.DataFrame(data, index=['a', 'b', 'c', 'd'])
    # 转置
    print(df.T)
    # 返回包含行索引列表和列索引列表的列表
    print(df.axes)
    # 返回每列的数据类型
    print(df.dtypes)
    # 判断DataFrame是否为空，True表示为空，False表示不为空
    print(df.empty)
    # ndim:返回维度数
    print(df.ndim)
    # 返回形状
    print(df.shape)
    # 返回DataFrame的元素总数
    print(df.size)
    # 以numpy形式返回数组
    print(df.values)
    # 获取前n行，默认n=5
    print(df.head(3))
    # 获取后n行，默认n=5
    print(df.tail(3))


if __name__ == '__main__':
    test01()
