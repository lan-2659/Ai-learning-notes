from matplotlib import pyplot as plt
import numpy as np


def test01():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    fig, axs = plt.subplots()
    axs.plot(x, y)
    # grid():在图形中添加网格线
    # 参数：
    # b：是否显示网格线
    # which：网格线类型，可以是 'major'（主刻度）、'minor'（次刻度）或 'both'（主刻度和次刻度）。
    # axis：显示哪些的网格线，可以是'x'、'y'、'both'
    # **kwargs：可以设置color(网格线颜色)、linestyle(线条样式)、linewidth(线条粗细)
    plt.grid(True, which='both', axis='both', color='r', linestyle='-', linewidth=1)

    # xscale、yscale：设置x、y轴的刻度类型
    plt.xscale('linear')
    plt.yscale('linear')

    # set_xlim、set_ylim：设置x、y轴的取值范围
    axs.set_xlim(0, 5)
    axs.set_ylim(0, 1)

    # set_xticks、set_yticks：手动设置x、y轴的刻度列表
    axs.set_xticks([0, 2, 4, 6])
    axs.set_yticks([0, 0.4, 0.8, 1])

    plt.show()


def test02():
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    fig, axs = plt.subplots()
    axs.plot(x, y1)

    # twinx、twiny：共享坐标轴
    # axs1 = axs.twinx()
    #
    # axs1.plot(x, y2, 'r')

    axs2 = axs.twiny()
    axs2.plot(x, y2, 'g')

    plt.show()


if __name__ == '__main__':
    test02()
