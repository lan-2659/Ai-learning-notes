# json模块
"""
通过这个模块可以很容易的把Python数据结构转换为JSON格式的字符串
而且JSON格式存储的数据可被多种语言读取并不局限于Python
"""


import json  # 导入json模块
from pathlib import Path

numbers = [1, 2, 3, 4, 5]  # 要存储数据

path = Path("numbers.json")  # JSON格式数据只能存储在后缀为.json的文件中
contents = json.dumps(
    numbers
)  # 用json模块中的dumps()函数将Python对象(除了集合，转换集合会报错)转换成JSON格式字符串
path.write_text(contents)

content = path.read_text()  # 直接从json文件中读取的内容是str类型
print(type(content))
number = json.loads(content)  # 通过json模块中的loads()方法可以把读取的内容转换为原类型
print(number)
