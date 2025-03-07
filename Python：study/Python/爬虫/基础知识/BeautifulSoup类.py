from bs4 import BeautifulSoup


# 这个是专门处理 HTML 结构，和 XML 文档的

soup = BeautifulSoup(response.text, 'html.parser')    # 使用时需要指定用哪一个解析器，html.parser 是用于解析 HTML 格式内容的

for i in soup.find_all('h3'):       # find_all()方法会查找所有指定的内容并返回一个迭代器
    print(i.find('a').string)       # find()方法只会查找第一个指定的内容
# 查找成功后返回的内容是 HTML 格式的，如果只想看到字符串内容，可以加上 .string 后缀

soup.get_text()   # 返回一个字符串，这个方法会提取调用对象的所有文本内容
soup.get_text(strip = True)  # 返回一个字符串，这个方法会提取调用对象的所有文本内容，且会去掉所有空白字符

