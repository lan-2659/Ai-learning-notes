# cv2.VideoCapture方法  
"""  
cv2.VideoCapture 是OpenCV中用于捕获视频（来自文件或摄像头）的核心类
支持读取视频帧、设置视频属性等功能  


## 一、语法格式  
### 1. 初始化方法  
cap = cv2.VideoCapture(source)  


### 2. 主要参数  
| 参数     | 类型       | 说明                                                                 |  
|----------|------------|----------------------------------------------------------------------|  
| `source` | int/str     | 视频源（必选）：<br>- 整数`0`/`1`等：摄像头索引（通常0为默认摄像头）<br>- 字符串：视频文件路径（如"video.mp4"）                               |  


## 二、核心方法与属性  

### 1. 读取视频帧  
```python  
ret, frame = cap.read()  
```  
- **返回值**：  
  - `ret` (bool)：是否成功读取帧（`True`/`False`）  
  - `frame` (numpy.ndarray)：读取的视频帧（BGR格式，形状为H×W×3）  

### 2. 检查视频源是否打开  
```python  
cap.isOpened()  
```  
- 返回`True`：视频源成功打开（文件存在/摄像头可用）  
- 返回`False`：需检查路径/摄像头连接  

### 3. 设置视频属性（可选）  
```python  
cap.set(propId, value)  
```  
- **常用`propId`（属性标识符）**：  
  | 标识符                | 数值 | 说明                     | 示例（设置640x480分辨率）       |  
  |-----------------------|------|--------------------------|--------------------------------|  
  | `cv2.CAP_PROP_FRAME_WIDTH`  | 3    | 帧宽度                   | `cap.set(3, 640)`              |  
  | `cv2.CAP_PROP_FRAME_HEIGHT` | 4    | 帧高度                   | `cap.set(4, 480)`              |  
  | `cv2.CAP_PROP_FPS`        | 5    | 帧率                     | `cap.get(5)` 获取当前帧率       |  

### 4. 获取视频属性  
```python  
value = cap.get(propId)  
```  

### 5. 释放资源（必须！）  
```python  
cap.release()  
```  


## 三、注意事项  
1. **资源释放**：  
   - 无论读取视频文件还是摄像头，使用完毕必须调用`cap.release()`，避免内存泄漏  
   - 配合`cv2.destroyAllWindows()`关闭所有窗口  

2. **摄像头索引**：  
   - 单摄像头通常为`0`，多摄像头尝试`1`/`2`等（根据系统设备列表）  
   - 部分设备需用完整路径（如Linux的`/dev/video0`）  

3. **视频文件兼容性**：  
   - 支持常见格式（MP4、AVI等），解码依赖系统FFmpeg/OpenCV编译时的支持  
   - 读取失败时检查文件路径、编码格式（推荐使用H.264编码的MP4）  

4. **实时性处理**：  
   - 摄像头捕获时，`cap.read()`可能存在延迟，需配合`waitKey(1)`刷新窗口  


## 四、举例：实时摄像头捕获与视频文件读取  

### 案例1：调用摄像头实时显示（按键退出）  
```python  
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
```  

### 案例2：读取视频文件并逐帧处理  
```python  
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
```  


## 五、常见问题解决  
### 1. 摄像头无法打开  
- **原因**：索引错误、权限问题、设备被占用  
- **解决**：  
  - 尝试`cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)`（Windows DirectShow后端）  
  - 检查任务管理器中摄像头是否被其他程序占用  

### 2. 视频播放卡顿/不同步  
- **原因**：帧率设置不当、窗口刷新延迟不足  
- **解决**：  
  - 设置`waitKey(int(1000/帧率))`（如30fps用`waitKey(33)`）  
  - 降低分辨率（`cap.set(3, 640); cap.set(4, 480)`）  

### 3. 中文路径读取失败  
- **解决**：使用英文路径，或通过`cv2.imdecode`读取字节流（需先读取文件为二进制）  

通过`cv2.VideoCapture`，可轻松实现视频/摄像头的捕获与处理，核心流程为：**初始化→读取帧→处理帧→释放资源**。实际应用中需根据场景设置分辨率、帧率等属性，并注意资源释放以保证程序稳定性。
"""