# 输出全部关键字
import keyword

for key in keyword.kwlist:
    print(key)


# 关键字 'in'
"""
用于检查一个元素是否存在于某个序列中,存在返回True,不存在返回False
可用于 字符串、列表、元组、集合、字典

示例:
'a' in list_

# 如果不对字典进行处理，那么in只能对字典中的键使用
"""


# None 关键字
"""
Python中 None 表示空
"""


# lambda关键字
"""
用于创建一个匿名函数
格式: lanmbda 形参名：返回值
示例: 
square = lambda x: x ** 2
"""


# nonlocal 关键字
"""
nonlocal x  # 声明这个x是外层函数中的变量（如果外层中有多个x，会选择最近那个），如果外层函数中不存在x会报错
nonlocal 用于访问或修改一个外层函数中声明的变量
"""


# global 关键字
"""
global x    # 导入全局变量x，如果这个x不存在会报错
如果没有声明 global，你在函数中对全局变量赋值时会创建一个局部变量，而不是修改全局变量。
"""
