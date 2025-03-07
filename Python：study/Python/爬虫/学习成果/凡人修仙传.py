import requests
from fake_useragent import UserAgent as ua
from bs4 import BeautifulSoup
import json
from pathlib import Path

headers = {'user-agent': f'{ua.random}'}

# 向原始网页发送请求
url = 'https://yhdm.one/vod/2020929790.html'
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

url_list = []   # 保存地址
title_list = []     # 保存集数
# 获取每一集的网站地址，并标注好集数
for i in soup.find('div', class_="ep-panel mb-3").find_all('li'):
    url = 'https://yhdm.one' + i.find('a')['href']
    url = url.rstrip('.html')
    url = url.replace('vod-play', '_get_plays')
    title_list.insert(0, i.find('a')['title'])
    url_list.insert(0, url)
title_url = list(zip(title_list, url_list))     # 将地址和集数打包装好

num = 0
for i in title_url:
    res = requests.get(i[1], headers=headers)
    s = BeautifulSoup(res.text, 'html.parser')
    i += ('https://yhdm.one' + s.find('li').find('a')['href'],)
    i = list(i)
    i.pop(1)
    i = tuple(i)
    title_url[num] = i
    num += 1
    print(i)



