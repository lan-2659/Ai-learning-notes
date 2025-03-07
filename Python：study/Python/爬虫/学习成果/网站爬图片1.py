import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36 SLBrowser/9.0.3.5211 SLBChan/105'}
response = requests.get('https://dnjm5.com/manga-detail/52858', headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')
num = 1
for i in soup.find('ul', class_='content_playlist clearfix').find_all('li'):
    num_1 = 1
    url = i.find('a')['href']
    response_1 = requests.get(f'https://dnjm5.com{url}', headers=headers)
    soup_1 = BeautifulSoup(response_1.text, 'html.parser')
    for j in soup_1.find_all('img'):
        if j.get('data-original') == None:
            continue
        response_2 = requests.get(j.get('data-original'), headers=headers)
        with open(fr'D:\Python：study\爬虫\学习成果\图片\第{num}章 第{num_1}页.jpg', 'ab') as file:
            file.write(response_2.content) 
        num_1 += 1
        print(f'第{num}章 第{num_1}页 下载完成！')
    num += 1
    time.sleep(0.5)