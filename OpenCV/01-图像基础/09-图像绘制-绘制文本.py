# 使用cv2.putText()方法绘制英文文本
"""  
cv2.putText()

    语法格式：  
    cv2.putText(img, text, org, fontFace, fontScale, color, thickness=1, lineType=cv2.LINE_8, bottomLeftOrigin=False)  

        参数说明：  
        img: 输入图像（numpy数组，在该图像上绘制文本）  
        text: 要绘制的文本字符串（支持ASCII字符，中文需特殊处理，如使用PIL库或指定字体文件）  
        org: 文本基线的左下角坐标，格式为 (x, y)（整数坐标）  
        fontFace: 字体类型（OpenCV内置字体或TrueType字体，需通过标志指定）  
            常用内置字体标志：  
            - cv2.FONT_HERSHEY_SIMPLEX：正常大小无衬线字体  
            - cv2.FONT_HERSHEY_PLAIN：小号无衬线字体  
            - cv2.FONT_HERSHEY_SCRIPT_SIMPLEX：手写风格字体  
            - cv2.FONT_ITALIC：斜体标志（可与其他字体标志按位或组合，如 fontFace=cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_ITALIC）  
        fontScale: 字体缩放因子（浮点数，控制字体大小，如1.0表示默认大小）  
        color: 文本颜色，BGR格式（例如 (255,0,0) 表示蓝色）  
        thickness: 文本笔画粗细（默认值为1，必须为正整数或0，0的效果和1一样）  
        lineType: 线条类型（可选参数，默认值为cv2.LINE_8，支持cv2.LINE_AA抗锯齿）  
        bottomLeftOrigin: 用于控制坐标系（默认False，即y轴向上；True表示y轴向下，较少使用）  


    注意事项：
    无返回值，直接在输入图像 `img` 上绘制文本（原地修改）  
    字体限制：  
       - 仅支持OpenCV内置的有限字体（如HERSHEY系列），不支持直接使用TTF字体文件  
       - 若需绘制中文或复杂字体，需借助PIL库将文本渲染为图像后叠加到OpenCV图像上  


    举例：  
    import cv2  
    import numpy as np  

    # 创建512x512的黑色画布（BGR三通道）  
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)  

    # 绘制白色无衬线文本（默认字体，缩放因子1.2，线宽2，抗锯齿）  
    cv2.putText(canvas, "Hello, OpenCV!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)  

    # 绘制红色斜体文本（组合字体标志，缩放因子0.8，线宽1）  
    cv2.putText(canvas, "Italic Text", (50, 200), cv2.FONT_HERSHEY_PLAIN | cv2.FONT_ITALIC, 0.8, (0, 0, 255), 1)  

    # 绘制带抗锯齿的绿色大字体（缩放因子2.0，线宽3）  
    cv2.putText(canvas, "BIG TEXT", (50, 350), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2.0, (0, 255, 0), 3, cv2.LINE_AA)  

    # 计算文本尺寸并绘制边框（避免溢出）  
    text = "Dynamic Text"  
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)  
    cv2.rectangle(canvas, (50, 450 - baseline), (50 + text_width, 450 + text_height), (255, 0, 0), 1)  # 绘制文本区域边框  
    cv2.putText(canvas, text, (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 250), 2)  

    cv2.imshow("Text Demo", canvas)  
    cv2.waitKey(0)  
    cv2.destroyAllWindows()  
"""


# 使用PIL库和OpenCV结合绘制中文文本
"""
简述过程：
    先使用cv2.cvtColor()将OpenCV图像转换为RGB格式(灰度图就不用了)
    使用Image.fromarray()将OpenCV图像转换为PIL格式
    使用ImageDraw.Draw()创建画笔，并使用ImageFont.truetype()加载中文字体文件  
    使用draw.text()方法在PIL图像上绘制中文文本  
    将PIL图像转换为OpenCV格式:
        先使用np.array将PIL图像转换为numpy数组
        再使用cv2.cvtColor将numpy数组转换为OpenCV图像格式，需使用这个模式cv2.COLOR_RGB2BGR
    至此一张绘制好中文文本的图像就创建完成了  

    
# Image.fromarray()方法  
 
    语法格式：  
    Image.fromarray(array, mode=None)  

        参数说明：  
        array: 输入的numpy数组(表示图像数据，形状为(H, W, C)) 
            - 对于彩色图像：C=3（RGB格式，注意与OpenCV的BGR格式区分）  
            - 对于灰度图像：C=1或省略，数据类型通常为np.uint8（0-255）  
        mode: 可选参数，指定图像模式（自动推断时可省略）  
            - "L"：灰度图（单通道，array形状为H×W）  
            - "RGB"：三通道彩色图（array形状为H×W×3，值范围0-255）  
            - "RGBA"：四通道彩色图（含透明度，array形状为H×W×4）  


    注意事项：  
        返回一个PIL.Image对象（可用于后续绘图操作）


    举例：  
    import cv2  
    import numpy as np  
    from PIL import Image  

    # 1. 创建OpenCV格式的BGR图像（512x512黑色画布）  
    cv_img = np.zeros((512, 512, 3), dtype=np.uint8)  
    cv_img[:, :, 2] = 255  # 设置红色通道（BGR中的红色）  

    # 2. 转换为PIL的RGB图像  
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))  

    # 3. 验证图像模式（应为"RGB"）  
    print(pil_img.mode)  # 输出：RGB  



# ImageFont.truetype()方法  

    语法格式：  
    ImageFont.truetype(font=None, size=10, index=0, encoding="")  

        参数说明：  
        font: 必选参数，字体文件路径（.ttf/.otf格式）  
            - 支持本地字体文件路径（如"simhei.ttf"或绝对路径）  
            - 不支持字体名称（如"黑体"，必须用具体文件路径）  
        size: 字体大小（磅值，控制字符高度，常见12-48）  
        index: 可选参数，当字体文件包含多个字体时指定索引（默认0）  
        encoding: 可选参数，字体编码方式（默认空，自动检测）  


    注意事项：  
        返回一个ImageFont.FreeTypeFont对象（用于后续文本绘制）  
        font参数:
            (下面是Windows系统自带的支持中文字体路径，可以直接传入参数使用)
            C:\\Windows\\Fonts\\SimHei.ttf      (黑体)
            C:\\Windows\\Fonts\\msyh.ttf        (微软雅黑)


    举例：  
    from PIL import ImageFont  
     
    font = ImageFont.truetype("C:\\Windows\\Fonts\\SimHei.ttf", 30)  # 加载系统黑体字体  


# ImageDraw.Draw()方法  

    语法格式：  
    ImageDraw.Draw(im, mode=None)  

        参数说明：  
        im: 必选参数，目标PIL图像对象（将在该图像上绘制内容）  
        mode: 可选参数，指定绘图模式（默认与图像模式一致）  

        
    注意事项：  
        返回一个ImageDraw.Draw对象，可以进行文本、线条、形状等绘制操作  


    举例：  
    from PIL import Image, ImageDraw  

    pil_img = Image.new("RGB", (200, 100), (255, 255, 255))  # 创建白色画布（PIL图像）

    draw = ImageDraw.Draw(pil_img)  # 创建绘图对象 
 
 

# draw.text()方法  
 
    语法格式：  
    draw.text(xy, text, font=None, fill=None, anchor=None, spacing=0, align="left", direction=None)  

        参数说明：  
        xy: 必选参数，文本左上角坐标(x, y)  
        text: 必选参数，要绘制的文本字符串（支持Unicode字符，如中文、英文、符号混合）  
        font: 可选参数，ImageFont对象（指定字体样式和大小，不指定则使用默认字体）  
        fill: 可选参数，文本颜色（与图像模式一致，RGB图像传入(R, G, B)，灰度图传入0-255）  
        anchor: 可选参数，文本定位锚点（默认"left"，支持"center"/"right"等）  
        spacing: 可选参数，行间距（多行文本时使用，单位像素）  
        align: 可选参数，水平对齐方式（"left"/"center"/"right"，默认左对齐）  
        direction: 可选参数，文本方向（暂不常用，支持"ltr"/"rtl"/"ttb"）  


    注意事项：  
        无返回值，直接在创建这个绘制对象的图像上绘制文本（原地修改）
        PIL坐标系: 左上角为原点，y轴向下(与OpenCV一致)

    举例：  
    from PIL import Image, ImageDraw, ImageFont  

    pil_img = Image.new("RGB", (300, 150), (255, 255, 255))  # 准备PIL图像和绘图对象
    draw = ImageDraw.Draw(pil_img)  
    font = ImageFont.truetype("C:\\Windows\\Fonts\\SimHei.ttf", 24)  
      
    draw.text((50, 30), "你好，Pillow！", font=font, fill=(255, 0, 0))  # 绘制中文文本（左上角坐标(50, 30)，红色）

    pil_img.show()  # 显示图像

    

# 使用PIL库和OpenCV结合绘制中文文本的总体案例：

    import cv2  
    from PIL import Image, ImageDraw, ImageFont  
    import numpy as np  

    # ---------------------- 步骤1：读取图片 ----------------------  
    img = np.fromfile("OpenCv/images/1.jpg", dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    # ---------------------- 步骤2：设置中文文本参数 ----------------------  
    text = "你好，世界！\nHello, World!"           # 支持换行（\n）  
    font_path = "C:\\Windows\\Fonts\\SimHei.ttf"  # 使用系统自带的黑体字体
    font_size = 30                                # 字体大小
    x, y = 50, 100                                # 文本左上角位置  
    color = (255, 255, 0)                         # 黄色（RGB格式：红=255，绿=255，蓝=0）

    # ---------------------- 步骤3：用PIL绘制中文到图像上 ----------------------  
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 转换OpenCV图像（BGR）为PIL图像（RGB）
    draw = ImageDraw.Draw(pil_img)                                   # 创建绘制对象
    font = ImageFont.truetype(font_path, font_size)                  # 加载中文字体  
    draw.text((x, y), text, font=font, fill=color)                   # 绘制文本

    # 转换回OpenCV的BGR格式  
    img_with_chinese = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # ---------------------- 步骤4：显示结果 ----------------------  
    cv2.imshow("中文文本示例", img_with_chinese)  
    cv2.waitKey(0)  
    cv2.destroyAllWindows()
"""
