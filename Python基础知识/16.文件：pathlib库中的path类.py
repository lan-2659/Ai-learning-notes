from pathlib import Path  # 导入pathlib模块中的Path类

path = Path(
    "D:\Python：study\基础语法学习\pi_digits.txt"
)  # 创建Path类的实例，这个实例创建后会固定，之后再调用时不会发生变化

contents = path.read_text().rstrip()  # read_text()会读取并返回文件中的全部内容

print(contents)  # 可以使用rstrip()方法删掉右边的所有空白字符

print(type(contents))  # 使用read_text()方法返回的是一个str类型

lines = contents.splitlines()  # 对读取到的文件进行按行分割，并返回一个列表

pi_string = ""
for line in lines:
    pi_string += line.strip()
print(pi_string)
print(len(pi_string))

"""
birthday = input('请输入你的生日：')
if birthday in pi_string:
    print('你的生日在圆周率中。')
else:
    print('你的生日不在圆周率中。')
"""

path = Path("基础语法学习\porgramming.txt")
path.write_text(
    "I love programming!"
)  # 向这个路径写入数据，如果这个路径不存在，会先创建再写入，而且写入的内容会覆盖原有内容
print(path.read_text())
