# 字典是一种映射类型，字典用"{ }"标识，它是一个无序的“键(key) : 值(value)”对集合。
# “键(key)”必须使用不可变类型
# “键(key)”必须是唯一的，但“值(value)”是可以重复的。


# 创建字典
"""
dict_x = {}           # 空字典
dict_x = dict()       # 空字典

dict_x = dict(one=1, two=2, three=3)       #使用了dict方法创建了一个字典,使用此方法时'='前的部分不加任何引号，且前面的部分会被当做字符串使用（前面部分的命名必须符合变量的命名规则）
        # dict()函数可以传入可迭代对象，但是可迭代对象的每一个元素中必须包含两个数据，前者为键， 后者为值
        #    比如：[('one', 1), ('two', 2), ('three', 3), ('four', 4)]

dict_x = {"Alice": "2341", "Beth": "9102", "Cecil": "3258"}      # 直接将创建好的字典赋值给变量
"""


# 字典索引
"""
dict_x = {"Alice": "2341", "Beth": "9102", "Cecil": "3258"}

print(dict_x["Alice"])               # 输出Alice键对应的值，若这个键不存在就会报错
print(dict_x.get("Alice", 100))      # 输出Alice键对应的值，若这个键不存在会返回100(返回值可以自己设置，默认为None)

dict_x["Alice"] = "2341"             # 修改'Alice'键对应的值，若这个键不存在则会为字典新添上这个键

del dict_x['Jack']                   # 删除dict_x中的Jack键值对,若这个键不存在会报错

del dict_x                           # 删除dict_x这整个字典

print(dict_x)                        # 输出字典
"""


# 字典的常用API
"""
dict_x.items()                       # 返回一个可迭代对象，包含字典中的所有键值对

list(dict_x)                         # 返回一个由dict_x中的键组成的列表，注意不是键值对

dict_x.values()                      # 返回一个可迭代对象，包含字典的所有值

dict_x.keys()                        # 返回一个可迭代对象，包含字典的所有键

dict_x.clear()                       # 清空字典
"""


# 使用字典对象的update()方法可以将另一个字典的元素一次性全部添加到当前字典对象中，不返回值
# 如果两个字典中存在相同的“键”，则只保留另一个字典中的键值对
school1 = {"class1": 62, "class2": 56, "class3": 68, "class4": 48, "class5": 70}
school2 = {"class5": 78, "class6": 38}
school1.update(school2)
#'class5'所对应的值取school2中'class5'所对应的值78
# 此时school1={'class1': 62, 'class2': 56, 'class3': 68, 'class4': 48, 'class5': 78, 'class6': 38}


# 补充：字典中嵌套字典示例
cities = {
    "shanghai": {
        "country": "china",
        "population": 120_000_000_000,
        "fact": "no money no like",
    },
    "beijing": {
        "country": "china",
        "populaiton": 222_222_222_222,
        "fact": "no money no like",
    },
    "chendu": {
        "country": "china",
        "population": 333_333_333_333,
        "fact": "sexualal's tiantang",
    },
}
