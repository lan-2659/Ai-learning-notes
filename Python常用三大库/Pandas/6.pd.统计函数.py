import pandas as pd


def test01():
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': [100, 200, 300, 400, 500]
    }

    df = pd.DataFrame(data)
    # 统计每列的算数平均值
    print(df.mean())
    # 统计指定列的算数平均值
    print(df['A'].mean())
    # 指定多列，统计算数平均值
    print(df[['A', 'B']].mean())
    # 统计中位数
    print(df.median())
    # 统计每列的方差，默认是样本方差
    print(df.var())
    # 统计每列的标准方差
    print(df.std())


if __name__ == '__main__':
    test01()
