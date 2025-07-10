import pandas as pd
import numpy as np


def test01():
    data = {
        'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
        'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
        'C': [1, 2, 3, 4, 5, 6, 7, 8],
        'D': [10, 20, 30, 40, 50, 60, 70, 80]
    }

    df = pd.DataFrame(data)

    gdf = df.groupby('A')
    print(list(gdf))

    # get_group(): 根据指定的分组名称获取组内的数据
    df1 = gdf.get_group('bar')
    print(df1)

    mean = df.groupby('A')['C'].mean()
    print(mean)

    df['C_MEAN'] = mean
    print(df)

    # groupby()：分组
    # transform():用于在分组操作中对每个组内的数据进行转换，并将结果合并回原始的DataFrame
    # transform的参数形式：聚合函数的名称、通过numpy的聚合函数、lambda表达式自定义聚合函数
    # df['C_MEAN'] = df.groupby('A')['C'].transform('mean')
    # df['C_MEAN'] = df.groupby('A')['C'].transform(np.mean)
    df['C_MEAN'] = df.groupby('A')['C'].transform(lambda x: (x - x.mean()) / x.std())
    print(df)


if __name__ == '__main__':
    test01()
