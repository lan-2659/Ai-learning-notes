# random模块方法
"""
import random       # 导入随机数模块

num = random.randint(1,3)       # 这个方法会生成1-3之间的任意随机数（整形），包括1和3
print('生成随机数：{}'.format(num))

num_list = [1, 2, 3, 4, 5, 6]
random.choice(num_list)     # 这个方法会在传入的序列或者集合中随机选择一个元素返回

random.randrange(start, stop, step)     # 从指定范围内，按指定基数递增的集合中获取返回一个随机整数（start~stop，不包含stop）
                                        # start 默认为0， stop 必须填，step 默认为1

random.shuffle(list)    # 将序列的所有元素随机排序,修改原list，不返回值

"""


# time模块方法
"""
import time

now = time.time()   # 这个方法会返回当前时间，以秒的形式

time.sleep(3)       # 这个方法会让程序停止执行3秒

t1 = time.localtime()    # 返回当前的本地时间元组
   例如：(tm_year=2021, tm_mon=3, tm_mday=30, tm_hour=23, tm_min=18, tm_sec=22, tm_wday=1, tm_yday=89, tm_isdst=0)

time.strftime("%Y-%m-%d", t1)   # 格式化时间,如：'2021-03-30'
"""


# math模块
"""
import math

math.sqrt(x)        # 返回数字x的平方根。

math.pi             # 返回数学常量 π（圆周率）约等于 3.141592653589793

math.e              # 返回数学常量 e，即自然常数 2.718281828459045
"""


# collections模块方法
"""
import collections

x = (1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1)

x_dict = collections.Counter(x) 
    # 这个方法要求输入的对象是可哈希的（即必须是不可变类型，如整数、字符串、元组等），统计可迭代对象中每个元素个数后，返回一个字典 
"""


# shutil模块
"""
import shutil

shutil.copy(path1, path2)   # 将path1路径下的文件，复制到path2这个指定目录
"""
