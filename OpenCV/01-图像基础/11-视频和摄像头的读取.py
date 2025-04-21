"""  
cv2.VideoCapture 是OpenCV中用于捕获视频（来自文件或摄像头）的核心类
支持读取视频帧、设置视频属性等功能  


一、语法格式  
    cap = cv2.VideoCapture(source=None)  

        参数:
        source:(默认为None)  
            int: 摄像头索引（通常为0）  
            str: 视频文件路径（如"video.mp4"）

        注意事项：
        无返回值，但会实例化一个对象，用于后续操作
        如果传入路径就只能传入英文路径，不支持中文路径(opencv4.5+版本解决了这个问题)


二、实例对象的核心方法
cap.isOpened()    # 返回一个bool值，检查视频源或摄像头是否成功打开(存在) 

ret, frame = cap.read()  
  返回值：  
    ret(bool)：是否成功读取帧（`True`/`False`）  
    frame(numpy.ndarray)：读取的视频帧（BGR格式，形状为H×W×3）

cap.set(propId, value)    # 设置视频属性，会返回一个bool值，表示是否设置成功
cap.get(propId)           # 返回获取到的视频属性
  propId：  
      cv2.CAP_PROP_FRAME_WIDTH    获取或设置视频宽度  
      cv2.CAP_PROP_FRAME_HEIGHT   获取或设置视频高度  
      cv2.CAP_PROP_FPS            获取或设置视频帧率  
      cv2.CAP_PROP_FRAME_COUNT    获取视频总帧数  

cap.release()   # 释放资源，无返回值

ret = cap.open(source)  # 切换打开的视频文件或摄像头，返回一个bool值(表示是否成功)
            source:  
              int: 摄像头索引（通常为0）  
              str: 视频文件路径（如"video.mp4"）

ret = cap.open(apiPreference, source)              
            apiPreference: 可选，默认为cv2.CAP_ANY
              cv2.CAP_ANY: 默认值，自动选择合适的API
            source:



三、举例：实时摄像头捕获与视频文件读取  

案例1：调用摄像头实时显示（按键退出）  
  import cv2  

  # 1. 初始化摄像头（索引0）  
  cap = cv2.VideoCapture(0)  

  # 2. 检查是否成功打开  
  if not cap.isOpened():  
      print("错误：无法打开摄像头")  
      exit()  

  # 3. 设置分辨率（可选）  
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 宽度1280  
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 高度720  

  while True:  
      # 读取视频帧  
      ret, frame = cap.read()  
      if not ret:  
          print("错误：无法读取帧")  
          break  

      # 显示帧（窗口名"Camera"）  
      cv2.imshow("Camera", frame)  

      # 按键处理：按'q'退出  
      if cv2.waitKey(1) == ord('q'):  
          break  

  # 4. 释放资源  
  cap.release()  
  cv2.destroyAllWindows()  

案例2：读取视频文件并逐帧处理  
  import cv2  

  # 1. 初始化视频文件  
  cap = cv2.VideoCapture("input_video.mp4")  

  while cap.isOpened():  
      ret, frame = cap.read()  
      if not ret:  
          print("视频播放完毕")  
          break  

      # 示例：将帧转为灰度图  
      gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  

      # 显示灰度帧  
      cv2.imshow("Gray Video", gray_frame)  

      # 按键处理：按'q'退出  
      if cv2.waitKey(25) == ord('q'):  # 25ms延迟（匹配25fps）  
          break  

  cap.release()  
  cv2.destroyAllWindows()  


"""