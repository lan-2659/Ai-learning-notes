import pandas as pd


def test01():
    df = pd.DataFrame({
        "company": ['百度', '阿里', '腾讯'],
        "salary": [43000, 24000, 40000],
        "age": [25, 35, 49]
    })

    # 随机抽取行
    df1 = df.sample(n=2, axis=0)
    print(df1)

    # 随机抽取列
    df2 = df.sample(n=1, axis=1)
    print(df2)

    # 按比例随机抽取行
    df3 = df.sample(frac=0.3, axis=0)
    print(df3)


if __name__ == '__main__':
    test01()
