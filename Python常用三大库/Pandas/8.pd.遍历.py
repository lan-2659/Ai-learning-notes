import pandas as pd


# Series遍历
def test01():
    s = pd.Series([1, 2, 3, 4, 5])
    # 类似python list的方式遍历
    for i in s:
        print(i, end=' ')
    print('------------------')
    # items():返回Series对象索引和值
    for index, value in s.items():
        print(index, value)

    # 通过index遍历
    for index in s.index:
        print(index, s[index])

    # 通过values遍历
    for v in s.values:
        print(v)


# DataFrame遍历
def test02():
    data = {
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    }
    df = pd.DataFrame(data, index=['a', 'b', 'c'])
    # 遍历结果是列标签
    for i in df:
        print(i)

    # iterrows()：遍历行，返回行索引和行数据，行数据是Series对象
    for index, row in df.iterrows():
        print('------------')
        print(index)
        print(row)
    # itertuples(): 遍历行，返回包含行数据的元组，推荐使用
    # 参数：index：如果为False则不返回索引
    for row in df.itertuples(index=False):
        print(row)
        for i in row:
            print(i, end=' ')
        print()

    print('==========================')
    # items():遍历列,返回包含列索引和列数据的迭代器
    for col, values in df.items():
        print(col)
        print(values)


if __name__ == '__main__':
    test02()
