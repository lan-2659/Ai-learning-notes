import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def put_text(image, text, position, font_path, font_size, color):

    # 将 OpenCV 图像转换为 PIL 图像
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # 创建一个指定图象上的画笔
    draw = ImageDraw.Draw(pil_image)
    # 加载字体
    font = ImageFont.truetype(font_path, font_size)
    # 在图像上绘制文本
    draw.text(position, text, fill=color, font=font)

    # 将 PIL 图像转换为 ndarray 对象
    arr = np.array(pil_image)
    # 将 ndarray 对象转换为 OpenCV 图像
    image_with_text = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    return image_with_text


if __name__ == '__main__':
    image = cv2.imread("../../images/car3.png")

    # 文本
    text = "你好世界"
    # 起始文字的坐标
    start = (100, 200)
    # 字体
    font = "../myfont/simhei.ttf"
    # 字体大小
    size = 24
    # 颜色
    rgb = (255, 0, 0)

    # 画个文本
    img = put_text(image, text, start, font, size, rgb)
    cv2.imshow("images", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
