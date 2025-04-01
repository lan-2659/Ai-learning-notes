# 概念
"""
- NumPy 的全称是“ Numeric Python”，它是 Python 的第三方扩展包，主要用来计算、处理一维或多维数组
- 在数组算术计算方面， NumPy 提供了大量的数学函数
- NumPy 的底层主要用 C语言编写，因此它能够高速地执行数值计算
- NumPy 还提供了多种数据结构，这些数据结构能够非常契合的应用在数组和矩阵的运算上
"""


# 优点
"""
NumPy 可以很便捷高效地处理大量数据，使用 NumPy 做数据处理的优点如下：
    - NumPy 是 Python 科学计算基础库
    - NumPy 可以对数组进行高效的数学运算
    - NumPy 的 ndarray 对象可以用来构建多维数组
    - NumPy 能够执行傅立叶变换与重塑多维数组形状
    - NumPy 提供了线性代数，以及随机数生成的内置函数
"""


# 与python列表区别
"""
NumPy 数组是同质数据类型（homogeneous），即数据类型在创建数组时指定，并且数组中的所有元素都必须是该类型。

Python 列表是异质数据类型（heterogeneous），即列表中的元素可以是不同的数据类型。列表可以包含整数、浮点数、字符串、对象等各种类型的数据。

NumPy 数组提供了丰富的数学函数和操作，如矩阵运算、线性代数、傅里叶变换等。

Python 列表提供了基本的列表操作，如添加、删除、切片、排序等。
"""


# 安装
"""
全局安装：
    pip install numpy==1.26.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/

conda环境：
    conda install numpy=1.26.0 -c https://pypi.mirrors.ustc.edu.cn/simple/
"""

