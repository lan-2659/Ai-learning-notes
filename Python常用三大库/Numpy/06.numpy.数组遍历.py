# ndarray数组的遍历
"""
for循环遍历:

    语法结构：
    for 循环变量 in ndarray对象：
        循环体

    注意事项：
    for循环只能遍历数组的第一维度的所有元素
    for循环可以直接遍历一维数组
    想要遍历多维数组，需要嵌套使用for循环很麻烦

    举例：
    import numpy as np

    a = np.array([[1, 2, 3], [4, 5, 6]])
    for i in a:
        print(i)        # 二维数组直接使用for循环遍历，输出结果是数组中的每个一维数组

    
numpy.nditer()方法：

    语法结构：
    numpy.nditer(op, flags=None, op_flags=None, order='C')
    参数说明：
    op: 要迭代的数组（或多个数组组成的列表）
    flags: 控制迭代行为的标志（如 multi_index, external_loop 等）
    op_flags: 操作数（数组）的属性（如 readonly, readwrite）
    order: 遍历顺序（'C' 行优先，'F' 列优先）

    注意事项：
    返回一个 numpy.nditer对象 可以用于迭代
    flags、op_flags两个参数只能用关键字传参，且只接收列表形式的参数
    order='C'或order='F'两者的区别：
        都是从 (0, 0, 0) 开始取值（以三维数组举例）
        当order='C'时，最右侧维度变化最快
        当order='F'时，最左侧维度变化最快
    
    举例：
    import numpy as np

    arr = np.array([[1, 2], [3, 4]])
    with np.nditer(arr) as it:
        for element in it:
            print(element, end=' ')  # 输出: 1 2 3 4

    with np.nditer(arr, flags=['multi_index']) as it:
    for element in it:
        idx = it.multi_index
        print(f"索引 {idx}, 值 {element}")
    # 输出:
    # 索引 (0, 0), 值 1
    # 索引 (0, 1), 值 2
    # 索引 (1, 0), 值 3
    # 索引 (1, 1), 值 4

    arr = np.array([[1, 2], [3, 4]])
    with np.nditer(arr, op_flags=['readwrite']) as it:
        for element in it:
            element[...] *= 2  # 所有元素乘以2
    print(arr)
    # 输出:
    # [[2 4]
    #  [6 8]]

    a = np.array([1, 2])
    b = np.array([[10], [20]])

    with np.nditer([a, b, None]) as it:  # None 表示输出数组
        for x, y, z in it:
            z[...] = x + y  # 广播运算

    result = it.operands[2]
    print(result)
    # 输出:
    # [[11 12]
    #  [21 22]]

    arr = np.arange(12).reshape(3, 4)
    with np.nditer(arr, flags=['external_loop'], order='F') as it:
        for chunk in it:
            print("块:", chunk)
    # 输出:
    # 块: [0 4 8]
    # 块: [1 5 9]
    # 块: [2 6 10]
    # 块: [3 7 11]
"""
