import requests
from bs4 import BeautifulSoup


headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36 SLBrowser/9.0.3.5211 SLBChan/105'}
web = '/books_1/index.html'
response_ = requests.get(f'http://books.toscrape.com/catalogue/category{web}', headers=headers)
soup_ = BeautifulSoup(response_.text, 'html.parser')
book_name = []
book_price = []
for i in soup_.findAll('h3'):
    name = i.find('a')['title']
    book_name.append(name)
for i in soup_.findAll('div', class_='product_price'):
    price = i.find('p').string[2:]
    book_price.append(price)
for i in range(len(book_name)):
    print(f'书名：{book_name[i]} \n价格：{book_price[i]}')
    print()

for i in soup_.find('ul', class_='nav nav-list').find('ul').findAll('li'):
    web = i.find('a')['href'][2:]
    response = requests.get(f'http://books.toscrape.com/catalogue/category{web}', headers=headers)
    if response.ok:
        contents = response.text
        soup = BeautifulSoup(contents, 'html.parser')
        book_name = []
        book_price = []
        for i in soup.findAll('h3'):
            name = i.find('a')['title']
            book_name.append(name)
        for i in soup.findAll('div', class_='product_price'):
            price = i.find('p').string[2:]
            book_price.append(price)
        for i in range(len(book_name)):
            print(f'书名：{book_name[i]} \n价格：{book_price[i]}')
            print()
    else:
        print('请求失败')