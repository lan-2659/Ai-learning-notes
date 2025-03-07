import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

headers = {'User-Agent':f'{UserAgent().random}'}
url = 'http://www.aguxs.com/aguxs.asp?id=7509329'
response = requests.get(url, headers=headers)
response.encoding = 'utf-8'
# print(response.text)
# print(response.encoding)
# print(response.encoding)

soup = BeautifulSoup(response.text, 'html.parser')
text = soup.find('td', class_='content').get_text()
# unicode_text = text.decode('iso-8859-1')
# utf8_text = text.encode('gbk', 'replace')  

with open('text.txt', 'w', encoding='utf-8') as f:
    f.write(text)
