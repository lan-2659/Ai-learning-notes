import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from fake_useragent import UserAgent
from pathlib import Path
import csv

# 定义函数 get_taptap_reviews，用于爬取评论
def get_taptap_reviews(url, user_agent, max_reviews=300):
    # 设置请求头，模拟浏览器访问
    headers = {'User-Agent': user_agent}
    reviews = []
    review_stars = []
    review_star_ = [] 
    reviews_ = []
    page = 1
    # 循环直到获取到足够的评论或者没有更多的评论页面
    while len(reviews) < max_reviews:
        # 构建评论页面的URL
        review_url = f"{url}?page={page}"
        # 发送GET请求
        response = requests.get(review_url, headers=headers)
        # 解析HTML内容
        soup = BeautifulSoup(response.text, 'html.parser')
        # 查找所有的评论div
        review_divs = soup.find_all('div', class_='text-box__content')
        # 查找所有的评论星级div
        review_stars_divs = soup.find_all('div', class_='review-rate__highlight')
        # 如果没有找到评论或评论星级，则退出循环
        if len(review_divs) == 0 or len(review_stars_divs) == 0:
            break
            
        # 遍历每个评论div
        for review_div in review_divs:
            # 获取评论文本并去除前后空白
            review = review_div.text.strip()
            # 将评论添加到列表中
            reviews.append(review)

        # 遍历每一个评论星级div
        for review_star_div in review_stars_divs:
            # 获取星级
            review_star = int(review_star_div['style'][6:8]) / 18
            # 将星级添加到列表
            review_stars.append(int(review_star))

        # 翻页
        page += 1

    # 将星级4或5的打上标签1，星级1或2的打上标签0，丢弃全部星级为3的评论
    for i in range(len(review_stars)):
        if review_stars[i] == 5 or review_stars[i] == 4:
            review_star_.append(1)
            reviews_.append(reviews[i])
        elif review_stars[i] == 2 or review_stars[i] == 1:
            review_star_.append(0)
            reviews_.append(reviews[i])
            
    # 将评论打包成一个字典
    star_review = {
        'review': reviews_,
        'sentiment': review_star_
    }

    # 返回评论列表
    return star_review
 
# 定义函数 save_reviews_to_text_to_scv，用于将评论保存到 csv
def save_reviews_to_text_to_scv(star_review, filename='taptap.xlsx'):
    df = pd.DataFrame(star_review)
    # 为csv文件打上标签
    df.columns = ['review', 'sentiment'] 
    # 保存为 CSV 文件
    df.to_csv('taptap.csv', index=False)  

# 主程序入口
if __name__ == "__main__":
    # 目标网页URL
    url = "https://www.taptap.cn/app/247283/review"
    # 模拟浏览器的User-Agent
    user_agent = f'{UserAgent.random}'
    # 设置最大爬取评论数为30
    max_reviews = 10
    # 调用函数爬取评论
    star_review = get_taptap_reviews(url, user_agent, max_reviews)
    # 调用函数将评论保存到text文件
    save_reviews_to_text_to_scv(star_review)