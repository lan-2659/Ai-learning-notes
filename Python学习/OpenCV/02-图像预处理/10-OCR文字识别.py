import cv2
import paddlehub as hub

image = cv2.imread('../../img2.png')

# 加载模型
model = hub.Module(name='chinese_ocr_db_crnn_server')

# 读取图片的文字信息，images可以不用关键字传参，但是传入的必须是一个列表
info = model.recognize_text(images=[image])

print(info)
if len(info) == 0:
    print("没有数据")
else:
    for item in info[0]["data"]:
        print("文字", item["text"])







