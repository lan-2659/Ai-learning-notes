import pandas as pd


def test01():
    # 创建两个示例 DataFrame
    left = pd.DataFrame({
        'key': ['K0', 'K1', 'K2', 'K3'],
        'A': ['A0', 'A1', 'A2', 'A3'],
        'B': ['B0', 'B1', 'B2', 'B3']
    })

    right = pd.DataFrame({
        'key': ['K0', 'K1', 'K2', 'K4'],
        'C': ['C0', 'C1', 'C2', 'C3'],
        'D': ['D0', 'D1', 'D2', 'D3']
    })

    # merge():根据相同列名合并两个DataFrame
    # 参数：
    # left：左侧DataFrame
    # right：右侧DataFrame
    # on：指定连接的列名，如果不指定，则默认按照两个DataFrame中相同的列名进行合并
    # how：指定连接方式，inner-内连接(默认)，outter-外连接，left-左连接，right-右连接

    # 内连接
    df1 = pd.merge(left, right, on='key')
    print(df1)

    # 左连接
    df2 = pd.merge(left, right, on='key', how='left')
    print(df2)

    # 右连接
    df3 = pd.merge(left, right, on='key', how='right')
    print(df3)


if __name__ == '__main__':
    test01()
