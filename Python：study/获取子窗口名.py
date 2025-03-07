import win32gui


# 获取窗口句柄
def get_window_handle(title):
    hwnd = win32gui.FindWindow(None, title)
    if not hwnd:
        raise Exception(f"未找到标题为 '{title}' 的窗口")
    return hwnd



window_title = "新建分组1"

# 获取窗口句柄
hwnd = get_window_handle(window_title)

# 查找子窗口
child_hwnd = win32gui.FindWindowEx(hwnd, None, None, "子窗口标题")
if child_hwnd:
    print(f"找到子窗口句柄: {child_hwnd}")
else:
    print("未找到子窗口")