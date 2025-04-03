from matplotlib import pyplot as plt
import numpy as np


def test01():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    # plot():绘制图形
    # 参数：
    # format_String：设置曲线颜色、线条样式等
    plt.plot(x, y, 'r:')

    # 显示图形
    plt.show()


def test02():
    # figure():生成画布
    # 参数：
    # figsize：画布的宽和高，数据格式是元组，单位：英寸
    fig = plt.figure(figsize=(12, 8))

    # add_axes：画布生成绘图区域
    # 参数：
    # left、bottom、width、height：取值范围都是0到1之间
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    x = np.linspace(-10, 10, 100)
    y = x ** 2

    # 通过绘图区域对象绘制图形，使图形绘制在绘图区域中
    ax.plot(x, y)
    plt.show()


def test03():
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    x = np.linspace(-10, 10, 100)
    y = x ** 2

    # legend():设置图形的图例
    # 参数：
    # handles：设置图形实例，数据格式是列表
    # labels：设置图形实例的标签说明，数据格式是列标
    # loc：设置图例显示的位置

    # plot：返回对象是一个列表
    # line = ax.plot(x, y)
    # print(type(line))
    # ax.legend(handles=line, labels=['x function'])

    # 图例设置的第二种方式：(常用)
    # 1.在plot方法中添加label参数，设置图例说明
    # 2.调用legend方法使图例生效，legend方法可以不用设置handles和labels
    ax.plot(x, y, label='x^2函数')
    ax.legend(loc='upper center')
    plt.show()


if __name__ == '__main__':
    test03()
