import pandas as pd


def test01():
    df = pd.DataFrame({'id': [1, 2, 3], 'name': ['zhangsan', 'lisi', 'wangwu'], 'age': [20, 21, 22]})
    print(df)

    # DataFrame某一列的Series对象：标签名是DataFrame的行标签
    print(df['id'])
    print('------------------')
    # DataFrame某一行的Series对象：标签名是DataFrame的列标签
    print(df.iloc[0])


def test02():
    # 1.创建DataFrame空对象
    df = pd.DataFrame()
    print(df)

    # 2.通过列表创建DataFrame
    df = pd.DataFrame([1, 2, 3])
    print(df)
    # 指定列索引名称
    df = pd.DataFrame([1, 2, 3], columns=['A'])
    print(df)

    # 3.通过二维数组创建DataFrame
    data = [['A', 1], ['B', 2], ['C', 3]]
    # 指定列索引和行索引，指定列/行索引时注意索引元素个数要和数据的元素个数一致
    df = pd.DataFrame(data, columns=['name', 'age'], index=['a', 'b', 'c'])
    print(df)

    # 4.通过列表嵌套字典创建DataFrame
    # 如果字典中有字段缺失，则默认填充NaN(not a number)
    data = [{'name': 'zhangsan', 'age': 20}, {'name': 'lisi', 'age': 21, 'sex': '男'}]
    df = pd.DataFrame(data)
    print(df)

    # 5.通过字典嵌套列表创建DataFrame
    data = {'name': ['zhangsan', 'lisi'], 'age': [20, 30]}
    df = pd.DataFrame(data)
    print(df)

    # 6.通过Series对象创建DataFrame
    # DataFrame默认使用Series对象中的索引标签作为DataFrame的行索引
    data = {'name': pd.Series(['zhangsan', 'lisi'], index=['a', 'b']), 'age': pd.Series([10, 20], index=['a', 'b'])}
    df = pd.DataFrame(data)
    print(df)


if __name__ == '__main__':
    test02()
