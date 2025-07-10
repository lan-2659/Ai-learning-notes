from matplotlib import pyplot as plt
import numpy as np
import os


def np_bar():
    x = [1, 2, 3, 4, 5, 6]
    y = [100, 50, 200, 350, 240, 190]

    fig, axs = plt.subplots()

    # bar():柱状图
    # 参数：
    # x：x轴的数据，格式是列表
    # height：y轴的数据，格式是列表
    # width：柱子的宽度，范围在0-1之间
    # align：柱子的对齐方式，center、edge
    axs.bar(x, y, width=0.6, align='edge')

    plt.show()


def np_bar_v2():
    x = [1, 2, 3, 4, 5, 6]
    y1 = [30, 40, 25, 60, 18, 45]
    y2 = [10, 15, 9, 21, 23, 10]

    fig, axs = plt.subplots()
    # bar()的bottom属性：柱状图底部的位置，默认为0
    axs.bar(x, y1, color='red')
    axs.bar(x, y2, bottom=y1, color='blue')

    plt.show()


def np_bar_v3():
    # 数据
    categories = ['A', 'B', 'C', 'D']
    values1 = [20, 35, 30, 25]
    values2 = [15, 25, 20, 10]

    # 创建图形和子图
    fig, ax = plt.subplots()

    # 计算柱状图的位置
    x = np.arange(len(categories))
    width = 0.35

    # 绘制第一个数据集的柱状图
    ax.bar(x - width / 2, values1, width, color='skyblue', label='Values 1')

    # 绘制第二个数据集的柱状图
    ax.bar(x + width / 2, values2, width, color='lightgreen', label='Values 2')

    plt.show()


def np_hist():
    data = np.random.randn(1000)

    fig, axs = plt.subplots()

    axs.hist(data, bins=60)

    plt.show()


def np_pie():
    x = [20, 30, 25, 40]
    label = ['A', 'B', 'C', 'D']
    fig, axs = plt.subplots()
    # pie():绘制饼图
    # 参数：
    # x：数据数组
    # labels：数据对应的标签名称
    # autopct：是否自动显示百分比
    # startangle：饼图的起始位置
    axs.pie(x, labels=label, startangle=90, autopct='%1.2f%%')

    plt.show()


def np_scatter():
    x = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    data = [
        [120, 132, 101, 134, 90, 230, 210],
        [220, 182, 191, 234, 290, 330, 310],
    ]

    y0 = data[0]
    y1 = data[1]

    fig, axs = plt.subplots()

    # scatter():散点图，可以用来比较两个或多个数据集之间的关系
    # 参数：
    # x：x轴数据
    # y：y轴数据
    axs.scatter(x, y0, color='red')
    axs.scatter(x, y1, color='green')

    plt.show()


def np_imread():
    filepath = os.path.dirname(__file__)
    print(filepath)

    filepath = os.path.join(filepath, 'leaf.png')
    print(filepath)

    filepath = os.path.relpath(filepath)
    print(filepath)

    # imread():读取图片，生成多维数组
    data = plt.imread(filepath)

    print(data.shape)

    print(data)

    # data = data + 0.1
    print(data)

    # imshow():将数组显示为图片
    plt.imshow(data)

    plt.show()

    data1 = np.transpose(data, (2, 0, 1))

    for i in data1:
        plt.imshow(i)
        plt.show()

        # imsave(): 将数组保存为图片
        plt.imsave('leaf1.png', i)


if __name__ == '__main__':
    np_imread()
