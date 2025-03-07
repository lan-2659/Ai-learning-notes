import requests
from bs4 import BeautifulSoup
headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36 SLBrowser/9.0.3.5211 SLBChan/105'
    }
# headers 作为参数只修改了请求来源，将我们的代码伪装成网页请求

num = 0
while num < 251:
    url = f'https://movie.douban.com/top250?start={num}&filter='
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')


    for i in soup.findAll('div', class_='hd'):
        num += 1
        name = i.find('a').find('span').string
        print(f'NO{num} : {name}')
