# 文件路径的两种操作方式
"""
1.使用传统的os模块
2.使用pathlib库（Python3.4+ 引入的库，用于面向对象地处理文件系统路径）
"""


# 文件路径中的'.'导入
"""
.  :当前目录
.. :上级目录
只有这两个，不存在 '...'，但是允许 '../..'
"""


# os模块和os.path模块的使用
"""
import os   

    # os 模块
        os.remove(path)         # 删除文件，不返回值

        os.getcwd()             # 返回当前的工作目录，这是一个绝对路径的字符串

        os.listdir(path)        # 返回一个列表，包含这个路径下的所有文件名（带后缀的）

        os.rename(old_name, new_name)  # 为文件改名，文件名应该写全（包括后缀）


    # os.path 模块
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


# 创建路径对象
path = Path()                           # 这不是创建空对象，这个语句等价于 path = Path(".")
path = Path("D:/")                      # 创建Path类的实例（路径对象）,这个路径可以是不存在的(在创建对象时不会报错)
path = path / "data" / "report.txt"     # 可以使用 / 运算符来连接路径
print(path)                             # 输出这个对象中保存的路径


# 路径对象的属性
path.name            # 返回文件名(带后缀)
path.stem            # 返回文件名(无后缀)
path.suffix          # 返回文件后缀
path.parent          # 返回父目录的路径对象


# 路径对象的方法
path.exists()        # 返回True 或 False，判断路径是否存在

path.is_file()       # 返回True 或 False，判断是不是一个文件
path.is_dir()        # 返回True 或 False，判断是不是一个目录

path.mkdir()         # 创建一个空目录，如果目录已存在会报错
path.rmdir()         # 删除一个空目录，如果目录不存在或非空会报错

path.touch()         # 创建一个空文件(如果不加后缀，则会创建文本文档)，如果文件已存在就不执行任何操作
path.unlink()        # 删除一个文件(不管有没有内容)，如果文件不存在会报错

path.open("r", encoding=None)    # 与内置函数 open 的功能一致，传参一致(少了文件路径的显示传入)，也同样是与with搭配使用
使用示例:
with path.open("r", encoding="utf-8") as f:    # 打开文件，并使用with语句，自动关闭文件
    print(f.read())

path.stat()          # 返回一个命名元组，该对象包含文件或目录的详细信息，如文件大小、创建时间、修改时间等（时间以时间戳表示）

new_path = path.with_name("new_report.txt")  # 返回一个路径对象，与原路径相比只有文件名不同(仅创建路径对象，文件系统并没有改变)
path.rename(new_path)                        # 将原对象指向的文件移动到新路径(如果如果原对象和新对象只有文件名不同，那么该方法的操作结果就是给文件改了名字)
                                             # 注意：rename()方法会改变文件系统，且如果 原对象指向的文件不存在 或 新对象指向的文件已存在 会报错

path.iterdir()       # 返回一个迭代器，迭代器中包含当前路径下的所有文件和子目录的路径对象
                     # 使用path.iterdir()方法时，path中的路径必须是一个有效的目录路径，否则会报错

path.resolve()       # 返回一个绝对路径对象（即用原对象的绝对路径创建的路径对象）


# Path 模块的静态方法（工具）
Path.home()      # 返回一个路径对象，表示当前用户的主目录，例如："C:\\Users\\26595"路径创建的路径对象(在我的电脑上)

Path.cwd()       # 获取当前工作目录，返回一个路径对象
"""
