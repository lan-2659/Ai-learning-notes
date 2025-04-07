import pandas as pd


def test01():
    data = {
        'A': [1, 2, 2, 3],
        'B': [4, 5, 5, 6],
        'C': [7, 8, 8, 9]
    }

    df = pd.DataFrame(data)

    # drop_duplicates():去重
    # 参数：
    # subset：要去重的列名的列表
    # keep：如何保留重复项，first-保留第一条重复项(默认)，last-保留最后一条重复项，False-不保留重复项
    # inplace：指定是否在原地修改数据
    df1 = df.drop_duplicates()
    print(df1)

    df2 = df.drop_duplicates(['A', 'B'])
    print(df2)

    df3 = df.drop_duplicates(['A'], keep=False)
    print(df3)


if __name__ == '__main__':
    test01()
