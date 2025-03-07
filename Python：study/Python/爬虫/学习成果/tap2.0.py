import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By


# 定义函数 get_taptap_reviews，用于爬取评论
def get_taptap_reviews(number):
        # 目标网页URL
    url = f"https://www.taptap.cn/app/{number}/review"

    reviews = []
    review_stars = []
    review_star_ = [] 
    reviews_ = []

    # 指定 ChromeDriver 路径
    service = Service(r'C:\Program Files\Google\Chrome\Application\chromedriver-win64\chromedriver.exe')
    driver = webdriver.Chrome(service=service)
    # 打开目标网页
    driver.get(url)
    # 给网页一些时间加载
    time.sleep(3)
    # 模拟滚动页面，直到页面底部
    scroll_pause_time = 2  # 每次滚动后暂停的时间（秒）
    scroll_height = driver.execute_script("return document.body.scrollHeight")
    while True:
    # for i in range(5):
        # 滚动页面到底部
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause_time)
        
        # 检查页面是否已经到底部
        new_scroll_height = driver.execute_script("return document.body.scrollHeight")
        if new_scroll_height == scroll_height:
            break  # 如果页面高度没有变化，说明已经到底部，退出循环
        scroll_height = new_scroll_height

    # 获取评论
    review_divs = driver.find_elements(By.CLASS_NAME, 'text-box__content')  
    # 获取星级
    review_stars_divs = driver.find_elements(By.CLASS_NAME, 'review-rate__highlight')  

    # 遍历每个评论div
    for review_div in review_divs:
        # 获取评论文本并去除前后空白
        review = review_div.text.strip().replace("\n", "")
        # 将评论添加到列表中
        reviews.append(review)

    # 遍历每一个评论星级div
    for review_star_div in review_stars_divs:
        # 获取星级
        review_star_style = review_star_div.get_attribute('style')
        review_star = int(review_star_style[7:9]) / 18
        # 将星级添加到列表
        review_stars.append(int(review_star))

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

    df = pd.DataFrame(star_review)
    # 为csv文件打上标签
    df.columns = ['review', 'sentiment'] 
    # 保存为 CSV 文件
    df.to_csv(f'tap_app_序号_{number}_评论数_{len(reviews_)}.csv', index=False)  

    # 执行完操作后退出浏览器
    # driver.quit()
    # 返回评论列表
    return star_review
 
# 主程序入口
if __name__ == "__main__":
    # 目标tap_app序号
    number = 247283
    # 调用函数爬取评论
    star_review = get_taptap_reviews(number)
