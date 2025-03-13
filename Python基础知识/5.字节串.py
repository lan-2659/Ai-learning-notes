# bytes 是一种内置的数据类型，用于表示字节串。
# 字节串是一种不可变的序列，用于存储原始的二进制数据，它的每个元素都是一个 字节（一个 8 位的值，范围是 0 到 255）

# 创建字节串
"""
在字面量字符串(即不能在变量名前加)前加 b 或 B 表示这是一个字节串，而不是普通的字符串（str）

byte_data = b'hello'    # 使用 b 前缀创建字节串
print(byte_data)        # 输出: b'hello'

# 使用 B 前缀创建字节串（等效于 b，但更明确）
byte_data = B'hello'
print(byte_data)        # 输出: b'hello'


用bytes()函数创建字节串

# 创建一个空的字节串
byte_data = bytes()
print(byte_data)        # 输出: b''

# 创建一个指定长度的字节串，所有字节初始化为 0
byte_data = bytes(5)    # 创建一个包含5个零字节的字节串
print(byte_data)        # 输出: b'\x00\x00\x00\x00\x00'

# 使用可迭代对象（如列表）创建字节串
byte_data = bytes([65, 66, 67])  # 对应 ASCII 码 65 = 'A', 66 = 'B', 67 = 'C'
print(byte_data)        # 输出: b'ABC'

"""


# bytes 与 str 之间的相互转换
"""
bytes 和 str 之间可以通过编码（encode()）和解码（decode()）进行转换

# 从字符串创建字节串
s = "hello"
byte_s = s.encode('utf-8')  # 编码成字节串
print(byte_s)  # 输出: b'hello'

# 从字节串解码为字符串
decoded_s = byte_s.decode('utf-8')
print(decoded_s)  # 输出: 'hello'

注意使用 encode()、decode() 时需要指定编码或解码方式

"""