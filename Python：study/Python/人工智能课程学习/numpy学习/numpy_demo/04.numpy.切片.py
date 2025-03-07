import numpy as np


def np_slice():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(arr)
    print('-------------')
    # 按行切片
    arr1 = arr[0:2]
    print(arr1)
    # 按列切片
    arr2 = arr[::, 0:2]
    print(arr2)
    # 按行和按列切片
    arr3 = arr[0:2, 0:2]
    print(arr3)

    # ...:切片时保留所有行或所有列
    arr4 = arr[0:2, ...]
    print(arr4)
    arr5 = arr[..., 0:2]
    print(arr5)
    # 切片产生新数组是原数组的一个视图(浅拷贝),修改视图中的数据会影响原数组
    arr1[0][0] = 100
    print(arr1)

    print('----------------')
    print(arr)



if __name__ == '__main__':
    np_slice()
