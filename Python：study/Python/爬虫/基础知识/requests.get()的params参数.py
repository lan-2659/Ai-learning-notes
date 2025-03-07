

# url传参
'''
字符串被当做url提交时会自动进行url编码处理，由于一般来说同一个网站编码方式相同，编码后只有字符串部分不同
我们可以通过修改字符串部分，来达到获得不同url的效果

明文：学习
密文：%E5%AD%A6%E4%B9%A0

from urllib.parse import quote, unquote
quote('学习')          # 会返回字符串的密文
unquote('%E5%AD%A6%E4%B9%A0')       # 会返回字符串的明文
'''

# 以百度为例
import requests
from fake_useragent import UserAgent

# 百度搜索时会将直接输入的字符串进行url编码(多余部分已删除，只留下最重要的部分)
url_baidu = r'https://www.baidu.com/s?wd=%E5%AD%A6%E4%B9%A0'  #此处 wd 后面的是密文，事实上用明文效果相同

url = r'https://www.baidu.com/s?'   # 剩下部分用get方法中的params参数传入

headers = {'User-Agent': f'{UserAgent().random}'}

# 构建请求参数字典，明文密文传入都可以，效果相同
params = {'wd': '学习'}     # 一般需要传入多个请求参数时才会使用

# 传入参数较少可以直接对 url 进行操作
'''
name = '学习'
url = f'https://www.baidu.com/s?wd={name}'
'''

response = requests.get(url, headers=headers, params=params)    

print(response.url)     # 这样输出的url中带的是密文

print(response.status_code)


# 如果做一个交互页面接收用户的关键字，然后对 url 进行操作， 这就是一个百度关键字爬虫

