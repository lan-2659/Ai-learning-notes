import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time

path = Path(r"C:\Users\26595\Desktop\圣墟.txt")
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36 SLBrowser/9.0.3.5211 SLBChan/105'}
response = requests.get('https://www.duquanben.com/xiaoshuo/58/58058/', headers = headers)

if response.ok:
    soup = BeautifulSoup(response.text, 'html.parser')
    for i in soup.find('ul', class_ ='mulu_list').find_all('li'):
        contents = i.find('a')
        title = contents['title']
        with open(path, 'a') as file:
            file.write('\n')
            file.write(f'{title}\r\n')
        url = contents['href']
        response_neirong  = requests.get(f'https://www.duquanben.com{url}', headers=headers)
        soup_neirong = BeautifulSoup(response_neirong.text, 'html.parser')
        for i in soup_neirong.find('div', class_ = 'contentbox'):
            with open(path, 'a') as file:
                text = i.get_text(strip = True).rstrip('上一章返回目录下一章')
                text = text.replace('\xa0', '\n')
                file.write(f'{text}\n')
        print(f'{title} 下载完成！')
        time.sleep(0.01)
        
else:
    print('请求失败！')