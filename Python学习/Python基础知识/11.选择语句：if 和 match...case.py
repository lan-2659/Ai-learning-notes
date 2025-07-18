# if语句
"""
# 单分支语法结构

    if 条件:
        执行语句1
        执行语句2
        执行语句3
        ...


# 双分支语法结构

    if 条件：
        执行体1
    else:
        执行体2

        
# 三目运算符

    结果1 if 条件 else 结果2

    
# 多分支语法结构

    if 条件1：
        执行体1
    elif 条件2:
        执行体2
    elif 条件n:
        执行体n
    else:
        执行体n+1
    # 注意：在多分支结构中，条件是从上往下判断的，一旦有条件成立，就会结束判断，并执行该判断语句后的执行体，其他语句不再执行


# if语句实例：

    n = int(input("请输入数字:0-3)\n"))
    if n == 0:
        print("你输入的数字是0")
    elif n == 1:
        print("你输入的数字是1")
    elif n == 2:
        print("你输入的数字是2")
    elif n == 3:
        print("你输入的数字是3")
    else:
        print("你未按照要求输入")
"""



# match...case语句
"""
# 在 Python 3.10 及以上版本中引入了 match-case 语句（结构化模式匹配）

# 基础语法

    match 表达式:
        case 模式1:
            代码块1
        case 模式2:
            代码块2
        ...
        case _:  # 默认情况
            默认代码块

# 关键注意事项

    执行顺序：从上到下匹配，第一个匹配成功的分支会被执行。

    变量绑定：模式中的变量名（如 x, y）会绑定到匹配的值。

    通配符 '_'：

        表示“忽略此值”，不绑定变量
        可以用于忽略整体，或者整体的部分
        例如：
            case _:             # 忽略整体，不绑定变量
            case (_, 10, _):    # 匹配三元组，且第二个元素必须为 10，其他两个元素可以是任意值
    
    匹配与捕获：
        case (x, _, z):     # 匹配任意三元组，且捕获第一个元素和最后一个元素（可用x、z进行访问）
"""