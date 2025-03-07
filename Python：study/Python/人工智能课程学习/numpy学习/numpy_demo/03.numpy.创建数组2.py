import numpy as np


def np_empty():
    # empty():创建指定形状的数组，元素是随机值
    # 参数：
    # shape：指定数组的形状
    # dtype：元素的数据类型
    # order：指定元素的存储方式：C、F
    arr = np.empty((2, 3))
    print(arr)


def np_zeros():
    # zeros():根据指定的形状创建数组，数组的元素值为0
    arr = np.zeros((3, 3), dtype=np.int32)
    print(arr)


def np_ones():
    # ones(): 根据指定的形状创建数组，元素值为1
    arr = np.ones((3, 3))
    print(arr)


def np_full():
    # full()：根据指定的形状创建数组，自定义填充的元素值
    # 参数：
    # fill_value：要填充的元素值
    arr = np.full((3, 3), 2)
    print(arr)


def np_arange():
    # arange():根据步长创建一个等差数列的数组
    # 参数：
    # start：起始值
    # stop：终止值（不包含）
    # step：步长（等差数列中前后数组的差值），默认为1
    arr = np.arange(0, 10)
    print(arr)
    arr1 = np.arange(1, 10, 2)
    print(arr1)


def np_linspace():
    # linspace():根据指定要平均分的份数创建一个等差数列的数组
    # 参数：
    # start：起始值
    # stop：终止值（默认包含）
    # num：要分多少份
    # endpoint: 指定是否包含终止值，默认True，如果为True表示包含终止值，False表示不包含
    # retstep：是否返回步长，默认为False，如果为True表示返回数组和步长，False表示不返回步长，只返回数组
    arr, step = np.linspace(0, 10, 20, endpoint=False, retstep=True)
    print(step, arr)


if __name__ == '__main__':
    # np_empty()
    # np_zeros()
    # np_ones()
    # np_full()
    # np_arange()
    np_linspace()
