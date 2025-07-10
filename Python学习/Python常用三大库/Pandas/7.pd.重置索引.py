import pandas as pd


def test01():
    data = {
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    }

    df = pd.DataFrame(data, index=['a', 'b', 'c'])

    # 调整行索引的顺序
    df1 = df.reindex(index=['c', 'a', 'b'])
    print(df1)
    # 在原来行索引基础上添加一个行索引，添加的行索引对应的数据默认为NaN
    df2 = df.reindex(index=['a', 'b', 'c', 'd'])
    print(df2)
    # 添加新的行索引，新索引会将原来索引对应的数据覆盖掉，以Nan填充
    df3 = df.reindex(index=['e', 'f', 'g'])
    print(df3)

    # 调整列的顺序
    df4 = df.reindex(columns=['C', 'A', 'B'])
    print(df4)
    # 在原来列标签的基础上添加新列
    df5 = df.reindex(columns=['A', 'B', 'C', 'D'])
    print(df5)
    # 如果用新列完全替代原来的列索引，则数据被修改NaN
    df6 = df.reindex(columns=['E', 'F', 'G'])
    print(df6)

    # 在原来行索引基础上添加新行，默认填充NaN，可以使用method进行Nan值的填充，ffill-前向填充，bfill-后向填充
    df7 = df.reindex(index=['a', 'b', 'c', 'd'], method='ffill')
    print(df7)

    # 新行的值可以使用fill_value自定义数值进行填充
    df8 = df.reindex(index=['a', 'b', 'c', 'd'], fill_value=0)
    print(df8)


def test02():
    df1 = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    }, index=['a', 'b', 'c'])

    df2 = pd.DataFrame({
        'A': [7, 8, 9],
        'B': [10, 11, 12]
    }, index=['b', 'c', 'd'])

    # 行索引对齐，df1以df2作为参照，来对齐行索引（列索引一致的）
    df3 = df1.reindex_like(df2)
    print(df3)


# df1和df2行索引一致，对齐列索引
def test03():
    df1 = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    }, index=['a', 'b', 'c'])

    df2 = pd.DataFrame({
        'B': [7, 8, 9],
        'C': [10, 11, 12]
    }, index=['a', 'b', 'c'])

    df3 = df1.reindex_like(df2)

    print(df3)


def test04():
    df1 = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    }, index=['a', 'b', 'c'])

    df2 = pd.DataFrame({
        'B': [7, 8, 9],
        'C': [10, 11, 12]
    }, index=['b', 'c', 'd'])

    df3 = df1.reindex_like(df2)
    print(df3)


if __name__ == '__main__':
    test04()
