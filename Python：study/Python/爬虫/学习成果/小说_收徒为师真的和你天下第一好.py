import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import random
import re

headers_addr = [UserAgent().random, UserAgent().random, UserAgent().random, UserAgent().random, UserAgent().random, UserAgent().random]
headers = {'User-Agent': random.choice(headers_addr)}
url = 'https://www.zmyou.net/book/7594/1.html'

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

match = re.search('href="(.*?)"', str(soup.find("span", class_="right")))
print(match)
# print(re.search('href=(\w*)>', str(soup.find("span", class_="right"))))
# print(soup.find("span", class_="right"))
print(str(soup.find("span", class_="right")))