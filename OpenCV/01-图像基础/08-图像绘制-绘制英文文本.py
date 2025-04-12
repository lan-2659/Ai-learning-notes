import cv2

image = cv2.imread("../../images/car3.png")

# 文本，需要为英文，如果是中文则会输出多个'？'
text = "hhhh"
# 起始文字的坐标
start = (100, 200)
# 字体
font = cv2.FONT_HERSHEY_SIMPLEX
# 字体大小
size = 2
# 颜色
rgb = (255, 0, 0)
# 线条的宽度
t = 5

# 传入参数
cv2.putText(image, text, start, font, size, rgb, t)

cv2.imshow("images", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
