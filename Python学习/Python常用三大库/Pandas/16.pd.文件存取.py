import pandas as pd


def test01():
    data = {
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Los Angeles', 'Chicago']
    }

    df = pd.DataFrame(data)

    # to_csv()：写文件
    # 参数：
    # path：文件路径，可以是绝对路径，也可以是相对路径
    # index：如果为False则不保存DataFrame的行索引，默认为True
    df.to_csv('pandas_test.csv', index=False)

    # read_csv()：读文件
    # 参数：
    # path：要读取文件的路径，可以是绝对路径，也可以是相对路径
    df1 = pd.read_csv('pandas_test.csv')
    print(df1)


if __name__ == '__main__':
    test01()
