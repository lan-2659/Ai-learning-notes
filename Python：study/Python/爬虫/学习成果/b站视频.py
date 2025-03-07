import requests
from bs4 import BeautifulSoup

headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36 SLBrowser/9.0.3.5211 SLBChan/105'}
response = requests.get('https://www.bilibili.com/video/BV1UT4y1b7Th/?p=8&spm_id_from=pageDriver&vd_source=7842b28185986074438400ce1572c1f7', headers=headers)
print(response.content)
if response.ok:
    soup = BeautifulSoup(response.text, 'html.parser')
    # print(soup)
    # print(soup.find('link'))
else:
    print('请求失败')

