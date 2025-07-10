import pandas as pd


def test01():
    data = {
        'B': [1, 2, 3],
        'A': [4, 5, 6]
    }

    df = pd.DataFrame(data, index=['b', 'a', 'c'])
    print(df)

    # sort_index():按照索引标签进行排序
    # 参数：
    # axis：排序的方向，0-按行，1-按列，如果axis为None，按行排序
    # ascending:表示是升序还是降序排序，True-升序，False-降序
    # inplace：表示是否在原地修改，True-在原数据上排序，False-生成副本，在副本上排序
    df1 = df.sort_index()
    print(df1)

    df2 = df.sort_index(axis=1)
    print(df2)

    df3 = df.sort_index(axis=0, ascending=False)
    print(df3)

    df.sort_index(axis=0, ascending=False, inplace=True)
    print(df)


def test02():
    data = {
        'A': [3, 2, 1],
        'B': [6, 5, 4],
        'C': [9, 8, 7]
    }
    df = pd.DataFrame(data, index=['b', 'c', 'a'])
    print(df)

    # sort_values():根据一列或多列进行排序
    # 参数：
    # by：要排序的列或者列的列表
    # ascending：升序或降序排序
    # inplace：是否在原地修改
    df1 = df.sort_values(by='A')
    print(df1)
    df2 = df.sort_values(by=['A', 'B'])
    print(df2)

    df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Age': [25, 30, 25, 35, 30],
        'Score': [85, 90, 80, 95, 88]
    })
    # 根据Age和Score排序，Age列按照降序排序，如果Age列中的数据相同，则按照Score列升序排序
    df1 = df.sort_values(by=['Age', 'Score'], ascending=[False, True])
    print(df1)


if __name__ == '__main__':
    test02()
