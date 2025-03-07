import requests     # 导入 requests 库，通过这个库可以发送各种HTTP请求

response = requests.get(url, headers, params)
# get函数用于发送HTTP GET请求，并返回一个Response类的实例，该对象包含了服务器响应的所有信息
'''
url 参数接收网站类型的字符串，注意传入的网址必须是完整的
headers 参数接收字典，字典中的内容主要用于伪装爬虫程序
params 参数接收字典，字典中的内容会附加在 url 后面
'''
 
url = "http://books.toscrape.com/"   # 这个网址是专门用于爬虫练习的网址，get()函数必须传入完整的网址 


