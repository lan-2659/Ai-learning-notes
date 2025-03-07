import time
import win32api
import win32con
import win32gui


# 获取窗口句柄
def get_window_handle(title):
    hwnd = win32gui.FindWindow(None, title)
    if not hwnd:
        raise Exception(f"未找到标题为 '{title}' 的窗口")
    return hwnd

# 将窗口坐标转换为屏幕坐标
def window_to_screen_coords(hwnd, x, y):
    rect = win32gui.GetWindowRect(hwnd)
    return (rect[0] + x, rect[1] + y)

# 发送鼠标点击事件
def send_mouse_click(hwnd, x, y):
    # 将坐标转换为屏幕坐标
    screen_x, screen_y = window_to_screen_coords(hwnd, x, y)
    
    # 发送鼠标按下和释放事件
    win32api.SendMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, win32api.MAKELONG(x, y))
    win32api.SendMessage(hwnd, win32con.WM_LBUTTONUP, 0, win32api.MAKELONG(x, y))

# 发送鼠标拖动事件
def send_mouse_drag(hwnd, start_x, start_y, end_x, end_y):
    # 发送鼠标按下事件
    win32api.SendMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, win32api.MAKELONG(start_x, start_y))
    
    # 发送鼠标移动事件
    win32api.SendMessage(hwnd, win32con.WM_MOUSEMOVE, win32con.MK_LBUTTON, win32api.MAKELONG(end_x, end_y))
    
    # 发送鼠标释放事件
    win32api.SendMessage(hwnd, win32con.WM_LBUTTONUP, 0, win32api.MAKELONG(end_x, end_y))

# 示例：在窗口内点击和拖动
def main():
    # 替换为你要操作的窗口标题
    window_title = "新建分组1"

    # 获取窗口句柄
    hwnd = get_window_handle(window_title)

    # 获取窗口的宽度和高度
    rect = win32gui.GetWindowRect(hwnd)
    window_width = rect[2] - rect[0]
    window_height = rect[3] - rect[1]

    # 点击
    send_mouse_click(hwnd, 263, 213)
    send_mouse_click(hwnd, 263, 213)
    send_mouse_click(hwnd, 263, 213)
    send_mouse_click(hwnd, 800, 212)
    send_mouse_click(hwnd, 800, 212)
    send_mouse_click(hwnd, 800, 212)
    send_mouse_click(hwnd, 1331, 209)
    send_mouse_click(hwnd, 1331, 209)
    send_mouse_click(hwnd, 1331, 209)

    # 示例：从窗口左上角滑动到右下角
    # send_mouse_drag(hwnd, 100, 100, window_width - 100, window_height - 100)

    # # 等待1秒
    # time.sleep(1)

    # 示例：在窗口中央再次点击
    # send_mouse_click(hwnd, window_width // 2, window_height // 2)

if __name__ == "__main__":
    main()