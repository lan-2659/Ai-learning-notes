import pandas as pd


def test01():
    data = {
        'A': [1, 2, 3, 4],
        'B': [5, 6, 7, 8],
        'C': [9, 10, 11, 12]
    }

    df = pd.DataFrame(data, index=['a', 'b', 'c', 'd'])
    print(df)

    # loc[]:根据行/列标签获取数据
    # 获取单行数据，返回结果是行的Series，该对象的索引是DataFrame的列索引
    print(df.loc['a'])
    # 通过切片获取多行数据，(切片包含终止值)，返回的结果是DataFrame
    print(df.loc['a':'c'])
    # 通过行标签和列标签获取数据,返回结果是一个标量
    print(df.loc['a', 'B'])

    print(df.loc['a':'c', 'B'])

    print(df.loc[['a', 'c'], ['A', 'C']])

    print(df.loc[df['B'] > 6, 'A'])


def test02():
    data = {
        'A': [1, 2, 3, 4],
        'B': [5, 6, 7, 8],
        'C': [9, 10, 11, 12]
    }

    df = pd.DataFrame(data, index=['a', 'b', 'c', 'd'])
    print(df)

    # iloc[]：根据数组下标获取数据，loc[]是根据索引标签获取数据
    # 获取单行数据
    print(df.iloc[1])

    # 通过切片获取多行数据,(不包含终止值)
    print(df.iloc[0:2])
    # 按行和列获取数据，结果是标量
    print(df.iloc[0, 0])


def test03():
    data = {
        'A': [1, 2, 3, 4],
        'B': [5, 6, 7, 8],
        'C': [9, 10, 11, 12]
    }

    df = pd.DataFrame(data, index=['a', 'b', 'c', 'd'])
    print(df)

    # DataFrame切片操作默认按行获取数据，但不能使用单标签或单下标获取
    # 根据数组下标切片
    print(df[0:2])

    # 根据索引标签切片
    print(df['a':'c'])

    # 切片错误操作
    # print(df['a'])
    # print(df['a':'c', 'B'])


'''
DataFrame的append方法在1.4版本后被弃用
'''


def test04():
    data = {
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    }
    df = pd.DataFrame(data, index=['a', 'b', 'c'])
    # 创建一个示例 Series
    new_row = pd.Series({'A': 7, 'B': 8}, name='d')

    # 使用 append 追加 Series
    df = df.append(new_row)

    print(df)


def test05():
    data = {
        'A': [1, 2, 3, 4],
        'B': [5, 6, 7, 8],
        'C': [9, 10, 11, 12]
    }

    df = pd.DataFrame(data, index=['a', 'b', 'c', 'd'])
    print(df)
    # 通过loc添加新行
    df.loc['e'] = [13, 14, 15]
    print(df)

    # 通过loc做修改行数据操作
    df.loc['a'] += 1
    print(df)
    df.loc['e'] = df.loc['d'] + 1
    print(df)


# concat():DataFrame连接
# 参数：
# objs：要连接的Dataframe的列表
# axis：连接方向：0-按行链接，1-按列连接
# ignore_index: 如果为True，则忽略原DataFrame的索引重新生成索引
# join：连接方式，outer-并集(默认)，inner-交集
def test06():
    data1 = {
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    }

    df1 = pd.DataFrame(data1)

    data2 = {
        'A': [7, 8, 9],
        'B': [10, 11, 12]
    }

    df2 = pd.DataFrame(data2)

    df = pd.concat([df1, df2], axis=0, ignore_index=True)
    print(df)

    df = pd.concat([df1, df2], axis=1)
    print(df)


def test07():
    data1 = {
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    }

    df1 = pd.DataFrame(data1)

    # data2 = {
    #     'A': [7, 8, 9],
    #     'B': [10, 11, 12],
    #     'C': [13, 14, 15]
    # }
    #
    # df2 = pd.DataFrame(data2)

    data2 = {
        'A': [7, 8, 9, 1],
        'B': [10, 11, 12, 1],
        'C': [13, 14, 15, 1]
    }

    df2 = pd.DataFrame(data2)

    df = pd.concat([df1, df2], axis=0, join='inner')
    print(df)

    df = pd.concat([df1, df2], axis=1, join='inner')
    print(df)


def test08():
    data = {
        'A': [1, 2, 3, 4],
        'B': [5, 6, 7, 8],
        'C': [9, 10, 11, 12]
    }

    df = pd.DataFrame(data, index=['a', 'b', 'c', 'd'])

    df1 = df.drop(['a'], axis=0)
    print(df1)
    # drop()：删除行或列
    # 参数：
    # index：要删除的行索引，格式是列表
    # columns：要删除的列索引，格式是列表
    # 如果指定了index或columns，labels会失效
    df2 = df.drop(index=['a'])
    print(df2)

    df3 = df.drop(columns=['C'])
    print(df3)


if __name__ == '__main__':
    test08()
