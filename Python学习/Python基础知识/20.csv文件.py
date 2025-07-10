"""
CSV（Comma-Separated Values）是一种简单的文本文件格式
用于存储表格数据，文件中的每行表示一行数据，字段之间用逗号分隔。

打开文件时建议明确指定 encoding 参数（如 utf-8）
"""

import csv  # Python 内置的 csv 模块非常方便处理 CSV 文件。

# 打开 CSV 文件
with open("example.csv", mode="r", encoding="utf-8") as file:
    reader = csv.reader(file)  # 创建 CSV 读取器
    for row in reader:
        print(row)  # 每行是一个列表

data = [["Name", "Age", "City"], ["Alice", 25, "New York"], ["Bob", 30, "Los Angeles"]]

with open("output.csv", mode="w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)  # 一次写入多行
