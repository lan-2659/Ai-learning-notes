import paddlehub as hub
import cv2
#识别图片的文字
def get_text(img):
    ocr = hub.Module(name="chinese_ocr_db_crnn_server")
    rs = ocr.recognize_text(images=[img])
    return rs

if __name__ =="__main__":
    img = cv2.imread(r"D:\G@T3N)}ZC)6H3SET$KQ8@GL.jpg")
    rs = get_text(img)
    print(rs)