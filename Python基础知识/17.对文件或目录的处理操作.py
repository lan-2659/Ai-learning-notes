# 文件路径的两种操作方式
"""
1.使用传统的os模块
2.使用pathlib库（Python3.4+ 引入的库，用于面向对象地处理文件系统路径）
"""


# os模块和os.path模块的使用
"""
import os

# os
os.remove(path)         # 删除文件，不返回值

os.getcwd()             # 返回当前的工作目录，这是一个绝对路径的字符串

os.listdir(path)        # 返回一个列表，包含这个路径下的所有文件名（带后缀的）

os.rename(old_name, new_name)  # 为文件改名，文件名应该写全（包括后缀）


# os.path
os.path.exists(path)        # 判断path指向的文件是否存在

os.path.basename(path)      # 去掉目录路径，返回文件名
os.path.dirname(path)       # 去掉文件名，返回目录路径

os.path.isdir(path)         # 判断路径是否存在且是一个目录
os.path.isfile(path)        # 判断路径是否存在且是一个文件

os.path.join(path1, path2, ..., pathN)  # 返回拼接好的路径(即根据操作系统选择合适的分隔符将传入的路径连接起来)
"""


# pathlib库的使用
"""
from pathlib import Path    # 导入pathlib模块中的Path类

path = Path("D:\Python：study\基础语法学习\pi_digits.txt")  # 创建Path类的实例（路径对象）
print(path)
"""
