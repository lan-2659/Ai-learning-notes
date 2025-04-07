import pandas as pd
import numpy as np


def test01():
    data = {
        'A': [1, 2, np.nan, 4],
        'B': [5, np.nan, np.nan, 8],
        'C': [9, 10, 11, 12]
    }
    df = pd.DataFrame(data)

    # fillna():空值填充，指定一个标量来填充DataFrame中的空值
    df1 = df.fillna(-1)
    print(df1)

    # dropna():删除DataFrame中的空值
    # 参数：
    # axis：指定删除的方向 0-按行(默认),1-按列
    df2 = df.dropna()
    print(df2)

    df3 = df.dropna(axis=1)
    print(df3)


if __name__ == '__main__':
    test01()
