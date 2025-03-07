import time
import pyautogui
import win32gui

# 获取窗口句柄
def get_window_handle(title):
    hwnd = win32gui.FindWindow(None, title)
    if not hwnd:
        raise Exception(f"未找到标题为 '{title}' 的窗口")
    return hwnd

# 获取鼠标相对于窗口左上角的坐标
def get_relative_mouse_position(hwnd):
    # 获取窗口的屏幕坐标
    rect = win32gui.GetWindowRect(hwnd)
    window_left, window_top = rect[0], rect[1]

    # 获取鼠标的屏幕坐标
    mouse_x, mouse_y = pyautogui.position()

    # 计算鼠标相对于窗口左上角的坐标
    relative_x = mouse_x - window_left
    relative_y = mouse_y - window_top

    return relative_x, relative_y

# 实时监听鼠标位置
def track_mouse_position(hwnd):
    try:
        while True:
            x, y = get_relative_mouse_position(hwnd)
            print(f"鼠标相对于窗口左上角的坐标: ({x}, {y})")
            time.sleep(0.1)  # 每 0.1 秒更新一次
    except KeyboardInterrupt:
        print("监听结束")

# 示例：监听鼠标在窗口中的位置
def main():
    # 替换为你要操作的窗口标题
    window_title = "新建分组1"

    # 获取窗口句柄
    hwnd = get_window_handle(window_title)

    # 开始监听鼠标位置
    print("开始监听鼠标位置，按 Ctrl+C 结束...")
    track_mouse_position(hwnd)

if __name__ == "__main__":
    main()