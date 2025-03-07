import pyautogui
import pygetwindow as gw
import time
from plyer import notification

# 设置延迟时间，避免操作过快
pyautogui.PAUSE = 1

# 获取指定窗口
def get_window_by_title(title):
    try:
        window = gw.getWindowsWithTitle(title)[0]  # 获取第一个匹配标题的窗口
        if window:
            return window
        else:
            raise Exception(f"未找到标题为 '{title}' 的窗口")
    except IndexError:
        raise Exception(f"未找到标题为 '{title}' 的窗口")

# 模拟鼠标点击（在窗口内）
def simulate_click_in_window(window, x, y):
    # 将窗口坐标转换为屏幕坐标
    screen_x = window.left + x
    screen_y = window.top + y
    pyautogui.moveTo(screen_x, screen_y)  # 移动鼠标到指定位置
    pyautogui.click()                     # 点击鼠标左键

# 模拟鼠标滑动（在窗口内）
def simulate_drag_in_window(window, start_x, start_y, end_x, end_y):
    # 将窗口坐标转换为屏幕坐标
    start_screen_x = window.left + start_x
    start_screen_y = window.top + start_y
    end_screen_x = window.left + end_x
    end_screen_y = window.top + end_y
    pyautogui.moveTo(start_screen_x, start_screen_y)  # 移动鼠标到起始位置
    pyautogui.dragTo(end_screen_x, end_screen_y, button='left')  # 按住左键拖动到结束位置

# 显示通知
def show_notification():
    notification.notify(
        title="脚本通知",  # 通知标题
        message="脚本运行结束！",  # 通知内容
        timeout=5  # 通知显示时间（秒）
    )

# 示例：在指定窗口内点击和滑动
def main():
    # 替换为你要操作的窗口标题
    window_title = "无标题 - 画图"
    
    # 获取窗口
    window = get_window_by_title(window_title)
    if not window:
        print(f"未找到标题为 '{window_title}' 的窗口")
        return
    
    # 激活窗口（将其置于最前面）
    window.activate()
    time.sleep(1)  # 等待窗口激活

    # 获取窗口的宽度和高度
    window_width = window.width
    window_height = window.height

    # 示例：在窗口中央点击
    simulate_click_in_window(window, window_width // 2, window_height // 2)

    # 示例：从窗口左上角滑动到右下角
    simulate_drag_in_window(window, 100, 100, window_width-100, window_height-100)

    # 等待1秒
    time.sleep(1)

    # 示例：在窗口中央再次点击
    simulate_click_in_window(window, window_width // 2, window_height // 2)

    # 显示通知
    show_notification()

if __name__ == "__main__":
    main()