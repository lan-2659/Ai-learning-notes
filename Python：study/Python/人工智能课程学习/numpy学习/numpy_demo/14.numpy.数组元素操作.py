import numpy as np


def np_resize():
    a = np.array([1, 2, 3, 4, 5])

    # np.resize(array, new_shape):数组形状修改，如果原数组中的元素不够，则重新遍历原数组的元素进行填充
    # 和reshape()区别：reshape要求新数组和原数组的元素总数相同；resize方法不受该约束
    print(np.resize(a, (3, 3)))


def np_append():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    # axis为None，返回的是一个一维数组，添加的值在数组尾部
    b = np.append(a, [7, 8, 9])
    print(b)
    # axis为0，按行添加数值到数组的尾部
    # 注意：values数组的维度数要和原数组的维度数一致，否则会抛异常
    # 在添加行的时候，values的列数要和原数组一致
    # 在添加列的时候，values的行数要原数组一致
    c = np.append(a, [[7, 8, 9]], axis=0)
    print(c)

    d = np.append(a, [[7], [10]], axis=1)
    print(d)


def np_insert():
    # insert():在原数组指定的索引前边插入元素
    # 注意：axis为None，返回的是一维数组
    # axis=0，按行插入
    # axis=1，按列插入
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.insert(a, 1, 100)
    print(b)

    c = np.insert(a, 1, 100, axis=0)
    print(c)

    d = np.insert(a, 1, 100, axis=1)
    print(d)

    e = np.insert(a, -1, 100, axis=1)
    print(e)


def np_delete():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    # np.delete(arr,obj,axis):按照指定的obj(索引)删除对应的元素
    # obj：要删除元素的索引，可以是标量，也可以是列表
    # axis=None，返回一维数组，然后删除对应索引的元素
    # axis=0，按行删除
    # axis=1，按列删除
    b = np.delete(a, 1)
    print(b)

    c = np.delete(a, 1, axis=0)
    print(c)

    d = np.delete(a, 1, axis=1)
    print(d)

    e = np.delete(a, [0, 1], axis=1)
    print(e)


def np_argwhere():
    a = np.array([[0, 1, 2], [3, 4, 5]])
    # np.argwhere(arr):默认返回数组中非0元素对应的索引下标
    # 也可以通过布尔索引获取满足布尔条件的元素对应的索引下标
    print(np.argwhere(a))

    print(np.argwhere(a > 4))


def np_unique():
    # a = np.array([1, 2, 2, 3, 4, 4, 5])
    # # np.unique():数组元素去重
    # # 参数：
    # # return_index:如果为True，返回新数组元素在原数组中索引位置
    # # return_inverse:如果为True，返回原数组元素在新数组中的索引位置，可以用来倒推原数组
    # # return_counts：如果为True，返回元素在数组中出现的次数
    # b, idx = np.unique(a, return_index=True)
    # print(b)
    # print(idx)
    #
    # c, idx = np.unique(a, return_inverse=True)
    # print(c)
    # print(idx)
    #
    # d, count = np.unique(a, return_counts=True)
    # print(count)

    # a = np.array([[2, 1, 8], [2, 1, 1], [2, 1, 1]])
    # c = np.unique(a, axis=0)
    # print(c)
    #
    # d = np.unique(a, axis=1)
    # print(d)

    arr = np.array([[[0, 1, 1],
                     [0, 1, 1],
                     [4, 5, 5]],

                    [[0, 1, 1],
                     [0, 1, 1],
                     [4, 5, 5]],

                    [[0, 1, 1],
                     [0, 1, 1],
                     [4, 5, 5]]])

    res1 = np.unique(arr, axis=0)
    print(res1)

    print('--------------------------')

    res2 = np.unique(arr, axis=1)
    print(res2)

    print('--------------------------')


    res3 = np.unique(arr, axis=2)
    print(res3)

if __name__ == '__main__':
    np_unique()
