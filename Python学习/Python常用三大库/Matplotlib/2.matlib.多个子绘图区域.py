from matplotlib import pyplot as plt
import numpy as np


def test01():
    # 准备数据
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.tan(x)

    # 生成figure画布
    fig = plt.figure(figsize=(12, 4))

    # 添加第一个绘图区域
    # ax1 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(131)
    ax1.plot(x, y1, label='sin(x)')
    # 使图例生效
    plt.legend()

    # 添加第二个子图区域
    # ax2 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(132)
    ax2.plot(x, y2, label='cos(x)')
    # 使图例生效
    plt.legend()

    # 添加第三个子图区域
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(x, y3, label='tan(x)')

    # 使图例生效
    plt.legend()

    plt.show()


def test02():
    # 准备数据
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.tan(x)

    # 按照nrows和ncols生成多个子图区域，在每个子图区域中绘制图形
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].plot(x, y1)
    axs[1].plot(x, y2)
    axs[2].plot(x, y3)

    plt.show()


def test03():
    # 准备数据
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.tan(x)

    # 生成一个子图区域，在一个子图区域中绘制多个图形
    fig, axs = plt.subplots()

    axs.plot(x, y1, 'r')
    axs.plot(x, y2, 'b')
    axs.plot(x, y3, 'g')

    plt.show()


def test04():
    # 准备数据
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.tan(x)
    y4 = np.exp(x)

    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 3), (1, 0))
    ax3 = plt.subplot2grid((3, 3), (1, 1))
    ax4 = plt.subplot2grid((3, 3), (2, 0))

    ax1.plot(x, y1)
    ax2.plot(x, y2)
    ax3.plot(x, y3)
    ax4.plot(x, y4)

    plt.show()


if __name__ == '__main__':
    test04()
