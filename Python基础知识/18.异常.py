# 异常的基本结构
"""
try:
    # 可能会引发异常的代码块，捕捉到异常后会根据其类型执行对应的except语句块
except 异常基本类型1 as e:       # as表示把捕获到的异常对象赋值给e
    # 异常处理代码块1
except 异常基本类型2:
    # 异常处理代码块2
except:     #如果不带任何类型，则会捕捉所有错误类型
    # 异常处理代码块
else:
    # 如果没有异常发生时执行的代码块
finally:
    # 无论是否有异常，都会执行的代码块，通常用于释放资源或清理操作
"""
# 注意这个结构是从上往下执行的，且只有一个except语句生效


"""
可以使用 'raise 异常类型' 这样的语句来主动引发异常
"""


"""
可以自己创建异常类
python 异常父类是 Exception 所有的基类是 object

class FileNotFoundError(Exception):
    def __init__(self):
        super().__init__("FileNotFoundError")   # 父类的初始化方法中方法可以不加任何参数

当直接输出这个异常的实例时，会将 super().__init__() 方法中传入的参数打包成一个元组输出     
"""
