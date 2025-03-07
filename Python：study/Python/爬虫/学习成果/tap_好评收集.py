import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

# 定义函数 get_taptap_reviews，用于爬取评论
def get_taptap_reviews(name, number):
        # 目标网页URL
    url = f"https://www.taptap.cn/app/{number}/review?mapping=%E5%A5%BD%E8%AF%84&label=0"

    reviews = []
    review_sentiment = [] 

    # 指定 ChromeDriver 路径
    service = Service(r'C:\Program Files\Google\Chrome\Application\chromedriver-win64\chromedriver.exe')
    driver = webdriver.Chrome(service=service)
    # 打开目标网页
    driver.get(url)
    # 给网页一些时间加载
    time.sleep(3)
    # 模拟滚动页面，直到页面底部
    # 每次滚动后暂停的时间（秒）
    scroll_pause_time = 2  
    scroll_height = driver.execute_script("return document.body.scrollHeight")

    # 循环收集1000个好评
    for _ in range(99):
        # 滚动页面到底部
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause_time)
        
        # 检查页面是否已经到最底部
        new_scroll_height = driver.execute_script("return document.body.scrollHeight")
        if new_scroll_height == scroll_height:
            break  
        scroll_height = new_scroll_height

    # 获取评论div
    review_divs = driver.find_elements(By.CLASS_NAME, 'text-box__content')  

    # 遍历每个评论div
    for review_div in review_divs:
        # 获取评论文本并去除前后空白
        review = review_div.text.strip().replace("\n", "")
        # 将评论添加到列表中
        reviews.append(review)
        # 为每一个好评打上标签
        review_sentiment.append(1)

    # 将评论打包成一个字典
    star_review = {
        'review': reviews,
        'sentiment': review_sentiment
    }

    df = pd.DataFrame(star_review)
    # 为csv文件打上标签
    df.columns = ['review', 'sentiment'] 
    # 保存为 CSV 文件
    df.to_csv(f'tap游戏_{name}_好评论数_{len(reviews)}.csv', index=False)  

    # 执行完操作后退出浏览器
    driver.quit()
    # 返回评论列表
    return star_review
 
# 主程序入口
if __name__ == "__main__":
    tap_game_names = ['阴阳师', '原神', '蛋仔派对', '王者荣耀', '第五人格', '英雄联盟手游', '我的世界_移动版', '和平精英', '龙族_卡塞尔之门', '地下城与勇士_起源', '心动小镇', '明日之后', '明日方舟', '金铲铲之战', '永劫无间', '火影忍者', '三国杀', '植物大战僵尸2', '地铁跑酷', '炉石传说']
    tap_game_numbers = [12492, 168332, 206776, 2301, 49995, 176911, 43639, 70056, 382099, 151294, 45213, 59520, 70253, 176937, 229966, 2247, 7035, 54031, 4, 213]
    for i in range(len(tap_game_names)):
        name = tap_game_names[i]
        # 目标tap_app序号
        number = tap_game_numbers[i]
        # 调用函数爬取评论
        star_review = get_taptap_reviews(name, number)
        print(f'已完成第 {i + 1} 个：{name}')
