# Python在1989年圣诞节期间被一个荷兰人创造（吉多.范罗苏姆）
# 于1991年被真正完成
"""
1 Byte = 8 bit
1 KB = 1024 B
1 MB = 1024 KB
1 GB = 1024 MB
1 TB = 1024 GB
"""
# 计算机的语言发展：机器语言-->汇编语言-->高级语言：对人的思维进行模仿，使用易懂的英文进行编程，此时的语言与底层交互不密切，可移植性非常强


# 注释     
"""
单行注释：
    # 以符号 "#" 开头的语句
单行注释快捷键：Ctrl + /

多行注释：
    '''
    像这样被三个引号包围起来的语句(可以是单引号双或引号，一般用双引号)
    '''

注：
1、注释是给人看的，注释是一种机器不会执行的代码
2、如非必要，多行注释中不要嵌套多行注释
3、养成做注释的好习惯，且注释要有明确的意义
"""


"""
Python中不能指定变量类型,为变量指定类型会报错
关键字：计算机中已经被定义好的，有特殊含义的单词

标识符：由用户自己定义的有特殊意义的符号
注：
1、标识符不能以数字开头
2、标识符只能由 a-z A-Z 0-9 _ 组成(即字母、数字、下划线)，其它字符都是非法的
    但是严格意义上来说，字母可以是符合unicode编码的任意字符，比如汉字，但是不建议这样做
    这是因为大家都用英文编码，所以字母一般都默认为英文字母
3、Python中严格区分大小写

Python中的命名规范：
1、小驼峰命名规则：helloWorld
2、大驼峰命名规则：HelloWorld
3、下划线命名规则：hello_world
如果是常量则需全部大写
"""


# Python中的API
"""
API（应用程序编程接口）: 指的是一组预定义的规则、函数、协议或工具，允许不同的软件组件之间相互通信和交互

Python中的API：
    Python 标准库或第三方库中提供的函数、类和方法的集合
    Web API（网络服务接口）
"""


# 语法糖的定义
"""
用更简洁的语法完成相同的逻辑，减少冗余代码。
例如：python中的列表推导式（简化列表生成逻辑，代码更加紧凑、简洁易懂）
"""


# Unicode（统一码）
"""
Unicode 是一种国际标准
旨在为世界上所有文字、符号和表情等字符分配一个唯一的数字编号（称为“码点”）
以便在不同计算机系统、编程语言和平台中实现一致的字符表示
"""


# 表达式
"""
代码中的一个片段，它通过一系列操作（如计算、函数调用等）最终会产生一个值
且表达式可以嵌套表达式
表达式的目的就是产生一个值
"""


# 语句
"""
代码中执行某个操作的完整指令，不直接返回值
语句分为空语句(用关键字pass表示)、单行语句、多行语句

语句中可以包含表达式，但表达式中不能包含语句
"""