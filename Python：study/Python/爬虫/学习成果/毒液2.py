import requests
from fake_useragent import UserAgent
from bs4 import BeautifulSoup

# 创建 UserAgent 实例
ua = UserAgent()
headers = {'User-Agent': ua.random}

# 发起请求
response = requests.get('https://www.tqys.tv/dy/dy4/duye2/bf-0-0.html', headers=headers)

# 检查响应状态
if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 查找包含视频信息的标签
    video_info = soup.find_all('span', string=lambda text: text and 'ffm3u8' in text)

    if video_info:
        for info in video_info:
            # 提取链接，假设链接在前一个标签中
            m3u8_link = info.find_previous('span').text  # 根据实际结构修改
            print(m3u8_link)
    else:
        print("没有找到视频信息。")
else:
    print('请求失败:', response.status_code)