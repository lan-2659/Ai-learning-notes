# 字典是一种映射类型，字典用"{ }"标识，它是一个无序的“键(key) : 值(value)”对集合。
# “键(key)”必须使用不可变类型
# “键(key)”必须是唯一的，但“值(value)”是可以重复的。


# 创建字典
"""
dict_x = {}           # 空字典
dict_x = dict()       # 空字典

dict_x = dict(one=1, two=2, three=3)       #使用了dict方法创建了一个字典,使用此方法时'='前的部分不能加任何引号
        # dict()函数需要传入可迭代对象，且可迭代对象的每一个元素中必须包含两个数据，前者为键， 后者为值
        #    比如：[('one', 1), ('two', 2), ('three', 3), ('four', 4)]

dict_x = {"Alice": "2341", "Beth": "9102", "Cecil": "3258"}      # 直接将创建好的字典赋值给变量
"""

# 字典的常用API
"""
print(dict1['Jack'])                              #输出Jack键对应的值

dict1['Jack'] = '1234'                            #为字典添加元素

dict1['Jack'] = '234'                             #修改'Jack'键对应的值

del dict1['Jack']                                #删除dict1中的Jack键值对

del dict1                                        #删除dict1这整个字典

dict1['Jack']                                    #返回Jack键对应的值，如果不存在Jack键会报错

dict1.get('Jack', '指定值')                        #返回Jack键对应的值，如果不存在Jack键会返回'指定值'，'指定值'可以不输入系统默认为None

dict1.items()                                    #返回一个键值对列表

list(dict1)                                      #返回一个由dict1中的键组成的列表，注意不是键值对

print(dict1.values())                            # 输出所有值

print(dict1.keys())                              # 输出所有键

print(dict1)                                     # 输出字典
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
