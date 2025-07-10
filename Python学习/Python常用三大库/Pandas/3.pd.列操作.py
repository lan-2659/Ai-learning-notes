import pandas as pd


def test01():
    df = pd.DataFrame({'name': ['zhangsan', 'lisi'], 'age': [20, 30], 'id': [1, 2]})
    # 根据列名获取某一列的数据，结果是列的Series对象
    s = df['name']
    print(s)
    # 将Series转换为python列表
    print(list(s))

    # 通过布尔索引获取数据
    df1 = df[df['age'] > 20]
    print(df1)

    # 添加一列空数据，在df中要指定新列名：df['sex']
    df['sex'] = None
    print(df)
    # 添加一列数据，数据的格式可以是python列表、Series对象等
    df['sex'] = ['男', '女']
    print(df)
    # assign(key=values)：添加新列，key作为Dataframe的列名，value作为DataFrame的列值，可以链式调用
    df = df.assign(address=['四川省', '重庆市']).assign(password=['12345', '12345'])
    print(df)

    # insert():在指定位置插入一个新列
    # 参数：
    # loc：新列要插入的位置
    # column：要插入的列名
    # value：要插入列值，可以列表、Series等
    df.insert(1, 'tel', ['13812345678', '17712345678'])
    print(df)


def test02():
    df = pd.DataFrame({'name': ['zhangsan', 'lisi'], 'age': [20, 30], 'id': [1, 2]})
    # 修改列，通过已存在的列名进行直接赋值即可修改该列的数据
    df['age'] = [40, 50]
    print(df)

    # 对某一列进行算术运算，然后再重新赋值给该列
    df['age'] = df['age'] - 10
    print(df)

    # 通过直接赋值修改列名
    df.columns = ['A', 'B', 'C']
    print(df)
    # 通过rename方法修改列名
    # 参数：
    # columns：指定新列名，格式：dict，dict中key是旧列名，value是新列名

    # columns属性和rename方法区别：columns在原数据上直接修改，rename是先备份一个副本，然后在副本上修改，不影响原数据
    df = df.rename(columns={'A': 'name', 'B': 'age', 'C': 'id'})
    print(df)

    print(df['age'].dtype)

    # astype():修改某一列的数据类型
    df['age'] = df['age'].astype('str')
    print(df['age'].dtype)


def test03():
    df = pd.DataFrame({'name': ['zhangsan', 'lisi'], 'age': [20, 30], 'id': [1, 2]})
    # drop():删除方法，既可以删除行，也可以删除列
    # 参数：
    # labels：要删除的列/行标签
    # axis：指定按行或按列删除，axis=0表示按行删除，axis=1表示按列删除
    # inplace：是否原地修改，如果为True则在原数据上进行删除，为False则先备份一个副本，然后在副本上进行删除，默认为False
    df1 = df.drop(['id', 'age'], axis=1)
    print(df1)
    print(df)

    df.drop(['id'], axis=1, inplace=True)
    print(df)


if __name__ == '__main__':
    test02()
