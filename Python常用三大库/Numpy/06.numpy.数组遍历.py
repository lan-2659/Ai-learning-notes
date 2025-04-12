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
    flags: 
        默认为 None，控制迭代行为的标志
        flags=['multi_index']: 将当前迭代到的元素的索引位置保存在 numpy.nditer对象 的 multi_index属性 中
        flags=['external_loop']: 对迭代器的每次迭代都返回尽可能大的连续内存块（而非逐元素遍历），会减少循环次数
    op_flags:
        默认为 readonly，操作数（数组）的属性
        op_flags=['readonly']: 只读
        op_flags=['readwrite']: 可读可改
        op_flags=['allocate']: 允许迭代器为 None 自动分配内存
    order: 遍历顺序（'C' 行优先，'F' 列优先），默认为'K'(即根据输入数组在内存中的存储顺序进行迭代)

    注意事项：
    返回一个 numpy.nditer对象(迭代器) 可以用于迭代
    如果传入了多个数组，对 numpy.nditer对象 迭代取值时，会先将这些数组广播成相同形状，然后根据order参数进行迭代取值(有几个数组就返回几个值)
    op 参数：
        传入多个数组时，这多个数组必须是形状相同的或者是可广播的
        传入多个数组时，可以使用 None 作为输出数组占位符（None：根据广播结果自动推断输出数组形状）
        （使用 None 时，必须将 None 对应的 op_flags 配置为 ['writeonly', 'allocate']）
    flags 与 op_flags 参数：
        只能用关键字传参，且只接收列表形式的参数 
        (注意传参方式: flags=['multi_index'] op_flags=['readwrite'])
        当 op参数 中传入多个数组 op_flags参数 的传参方式：
            np.nditer([a, b, None], op_flags=[['readonly'], ['readonly'], ['writeonly',  'allocate']]
    order='C'或order='F'两者的区别：
        都是从 (0, 0, 0) 开始取值（以三维数组举例）
        当order='C'时，最右侧维度变化最快
        当order='F'时，最左侧维度变化最快
    
    举例：
    import numpy as np

    arr = np.array([[1, 2], [3, 4]])
    with np.nditer(arr) as it:
        for element in it:              # 对 numpy.nditer对象 进行迭代时返回的每一个元素都是 0维数组视图
            print(element, end=' ')

    with np.nditer(arr, flags=['multi_index']) as it:   
    for element in it:
        idx = it.multi_index                   # multi_index属性 可以返回当前迭代到的元素的索引位置
        print(f"索引 {idx}, 值 {element}")
    # 输出:
    # 索引 (0, 0), 值 1
    # 索引 (0, 1), 值 2
    # 索引 (1, 0), 值 3
    # 索引 (1, 1), 值 4

    arr = np.array([[1, 2], [3, 4]])
    with np.nditer(arr, op_flags=['readwrite']) as it:      # 'readwrite': 可读可更改原数组元素
        for element in it:
            element[...] *= 2                   # 所有元素乘以2。注意：element *= 2 语句同样可以修改原数组元素
    print(arr)
    # 输出:
    # [[2 4]
    #  [6 8]]

    arr = np.arange(12).reshape(3, 4)
    with np.nditer(arr, flags=['external_loop'], order='F') as it:
        for chunk in it:                   # 每次迭代返回的是一个连续内存块
            print("块:", chunk)
    # 输出:
    # 块: [0 4 8]
    # 块: [1 5 9]
    # 块: [2 6 10]
    # 块: [3 7 11]

    # 传入多个数组与None占位符示例
    a = np.array([1, 2])
    b = np.array([[10], [20]])
    with np.nditer([a, b, None], op_flags=[['readonly'], ['readonly'], ['writeonly',  'allocate']]) as it:     
        for x, y, z in it:
            z[...] = x + y  
        result = it.operands[2]
    print(result)
    # 输出:
    # [[11 12]
    #  [21 22]]
"""
