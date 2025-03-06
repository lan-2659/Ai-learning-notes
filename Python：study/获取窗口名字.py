import pygetwindow as gw

# 获取所有窗口
windows = gw.getAllTitles()

# 打印所有窗口标题
for title in windows:
    if title:  # 过滤掉空标题
        print(title)