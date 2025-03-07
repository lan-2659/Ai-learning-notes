import numpy as np


def test01():
    # 创建numpy一维数组
    # dtype:指定数组元素的数据类型，可选参数
    # order:数据存储方式：C-按行，F-按列，可选参数
    # ndmin:指定数组的维度，可选参数
    arr = np.array([1, 2, 3, 4], dtype=int, order='F', ndmin=1)
    print(arr)
    print(type(arr))
    print(arr.dtype)
    print(arr.shape)

    arr.shape = (2, 2)
    print(arr.shape)
    print(arr)


def test02():
    # 创建二维数组
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    print(arr)
    print(arr.dtype)
    # ndim:数组的维度数
    print(arr.ndim)
    # shape:数组的形状，如：(2,3)表示2行3列
    # shape:除了可以获取数组形状，还可以设置数组形状，在原数组上修改形状，会影响原数组
    # shape在修改数组形状时，元素总数不能改变，如：一维数组元素个数为4，修改为二维数组时，可以是(2,2),但不能是(2,3)
    print(arr.shape)

    arr.shape = (3, 2)
    print(arr)

    # reshape():设置数组的形状，修改后的数组是原数组的一个视图，数据是共享的，修改新数组的元素值会影响元素组
    # shape参数：可以直接指定形状，也可以封装为一个元组作为参数
    # arr1 = arr.reshape(2, 3)
    arr1 = arr.reshape((2, 3))
    print(arr1)
    print(arr)
    # -1：形状的占位符，numpy根据指定和或列自动计算列或行
    arr2 = arr.reshape(-1, 2)
    print(arr2.shape)


# 入口方法
if __name__ == '__main__':
    test02()
