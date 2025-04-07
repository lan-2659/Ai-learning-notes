import pandas as pd
from matplotlib import pyplot as plt


def test01():
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 25, 30, 40],
        'C': [50, 60, 70, 80, 85]
    }
    df = pd.DataFrame(data)
    # pandas将matplotlib中的绘图方法统一封装为plot()方法
    # 参数：
    # kind: 指定图表类型的字符串，如：line、bar、hist、...

    # 如果要显示图片，还是需要调用matplotlib中的show()
    df.plot(kind='line')
    plt.show()

    df.plot(kind='bar')
    plt.show()

    df.plot(kind='hist')
    plt.show()

    df.plot(kind='scatter', x='A', y='B')
    plt.show()

    data = {
        'A': 10,
        'B': 20,
        'C': 30,
        'D': 40
    }
    series = pd.Series(data)
    # 绘制饼图
    series.plot(kind='pie', autopct='%1.1f%%')
    # 显示图表
    plt.show()


if __name__ == '__main__':
    test01()
