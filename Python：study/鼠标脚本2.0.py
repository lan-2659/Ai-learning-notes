from pywinauto import Application
import time

# 启动目标应用程序（如果未启动）
# 例如，启动记事本
app = Application(backend="win32").start(r"D:\leidian\ldmutiplayer\dnmultiplayerex.exe")

# 连接到目标窗口
window = app.window(title="雷电多开器")

# # 获取所有窗口
# windows = app.windows()

# # 打印每个窗口的标题
# for window in windows:
#     print(f"窗口标题: {window.window_text()}")

# 确保窗口存在
if not window.exists():
    print("未找到目标窗口")
    exit()

# 在后台模拟鼠标点击
def simulate_click_in_window(window, x, y):
    window.click_input(coords=(x, y))  # 在指定坐标点击

# 在后台模拟鼠标拖动
def simulate_drag_in_window(window, start_x, start_y, end_x, end_y):
    window.drag_mouse_input(
        press_coords=(start_x, start_y),  # 起始坐标
        release_coords=(end_x, end_y)     # 结束坐标
    )

# 示例：在窗口内点击和拖动
def main():
    # 获取窗口的宽度和高度
    rect = window.rectangle()
    window_width = rect.width()
    window_height = rect.height()

    # 示例：在窗口中央点击
    simulate_click_in_window(window, window_width // 2, window_height // 2)

    # 示例：从窗口左上角滑动到右下角
    # simulate_drag_in_window(window, 100, 100, window_width - 100, window_height - 100)

    # 等待1秒
    time.sleep(1)

    # 示例：在窗口中央再次点击
    # simulate_click_in_window(window, window_width // 2, window_height // 2)

if __name__ == "__main__":
    main()