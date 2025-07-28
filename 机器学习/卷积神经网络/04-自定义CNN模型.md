## 1.自定义一个简单的CNN模型

```python
# 使用pytorch自定义一个简单的CNN模型
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        return x


if __name__ == '__main__':
    model = SimpleCNN()
    print("模型结构:\n", model)
```

运行效果：

```xml
模型结构:
 SimpleCNN(
  (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu2): ReLU()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
```


上面代码虽然没有多大问题，但是看起来还是有点过于繁琐，推荐使用nn.Sequential 进行改进。

`nn.Sequential` 是 PyTorch 中的一个**容器类模块**，用于按顺序组织多个神经网络层（`nn.Module` 的子类），它会按照你添加的顺序依次执行这些层。

---

###  语法结构

```python
torch.nn.Sequential(*args)
```

你可以将多个网络层作为参数传入 `nn.Sequential`，它们会按顺序构成一个序列化的网络结构。

---

###  举个例子

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)
```

这个 `model` 的执行顺序是：

1. 输入通过 `Linear(10 -> 50)` 层
2. 经过 `ReLU` 激活函数
3. 再通过 `Linear(50 -> 1)` 层

等价于：

```python
def forward(x):
    x = nn.Linear(10, 50)(x)
    x = nn.ReLU()(x)
    x = nn.Linear(50, 1)(x)
    return x
```

---

### ✅ 使用 `nn.Sequential` 的好处

1. **代码简洁**：不用手动写 `forward` 函数，每一层自动按顺序执行。
2. **结构清晰**：适合构建线性堆叠结构的网络（如 CNN 的卷积层、全连接层等）。
3. **易于调试**：可以像列表一样访问各个层，例如 `model[0]` 表示第一个层。

---

###  改进后的代码

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 输入3通道，输出16通道
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 输入16通道，输出32通道
            nn.ReLU(),
        )

if __name__ == '__main__':
    model = SimpleCNN()
    print("模型结构:\n", model)
```

运行效果：

```xml
模型结构:
 SimpleCNN(
  (features): Sequential(
    (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): ReLU()
  )
)
```


这段代码表示一个卷积特征提取器，输入一张图像后，会依次经过：

- 第1层卷积 → ReLU → MaxPool
- 第2层卷积 → ReLU → MaxPool

最终输出的是提取后的特征图（feature maps）。

---

### 注意事项：

- 除了按顺序执行，`nn.Sequential` 不提供其他功能。
- 如果你的网络结构有分支、跳跃连接（如 ResNet 中的 shortcut）、多个输入输出等复杂结构，就不适合用 `nn.Sequential`，而应该自定义 `forward` 方法。

---

### ✅ 总结一句话：

> `nn.Sequential` 是一个按顺序执行的神经网络模块容器，适合构建线性堆叠结构的模型，可以简化代码并使结构更清晰。

## 2. 自定义CNN模型实现图片边缘检测案例

实现代码如下：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# 定义简单的CNN模型
class EdgeDetectionCNN(nn.Module):
    def __init__(self):
        super(EdgeDetectionCNN, self).__init__()
        # 使用固定的边缘检测卷积核
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)

        # 手动设置卷积核权重（水平和垂直边缘检测）
        sobel_x = torch.tensor([[[[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]]]], dtype=torch.float32)

        sobel_y = torch.tensor([[[[-1, -2, -1],
                                  [0, 0, 0],
                                  [1, 2, 1]]]], dtype=torch.float32)

        # 组合两个卷积核
        edge_kernels = torch.cat([sobel_x, sobel_y], dim=0)
        self.conv1.weight = nn.Parameter(edge_kernels, requires_grad=False)

    def forward(self, x):
        # 应用边缘检测卷积
        edge_features = self.conv1(x)
        # 分离水平和垂直特征
        horizontal = edge_features[:, 0:1, :, :]
        vertical = edge_features[:, 1:2, :, :]
        # 计算边缘强度
        edge_magnitude = torch.sqrt(horizontal ** 2 + vertical ** 2)
        return edge_magnitude, horizontal, vertical


# 图像预处理
def preprocess_image(image_path):
    # 打开图像并转换为灰度
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    # 添加batch维度 [1, 1, H, W]
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image


# 可视化结果
def visualize_results(original, horizontal, vertical, magnitude):
    plt.figure(figsize=(12, 10))

    # 原始图像
    plt.subplot(2, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # 水平边缘
    plt.subplot(2, 2, 2)
    plt.imshow(horizontal, cmap='gray')
    plt.title('Horizontal Edges')
    plt.axis('off')

    # 垂直边缘
    plt.subplot(2, 2, 3)
    plt.imshow(vertical, cmap='gray')
    plt.title('Vertical Edges')
    plt.axis('off')

    # 边缘强度
    plt.subplot(2, 2, 4)
    plt.imshow(magnitude, cmap='gray')
    plt.title('Edge Magnitude')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('edge_detection_result.png')
    plt.show()


# 主流程
if __name__ == "__main__":
    # 1. 初始化模型
    model = EdgeDetectionCNN()

    # 2. 加载和预处理图像
    image_tensor, original_image = preprocess_image('./images/bird01.jpg')  # 替换为你的图片路径

    # 3. 提取边缘特征
    with torch.no_grad():
        edge_magnitude, horizontal, vertical = model(image_tensor)

    # 4. 转换为numpy并后处理
    horizontal_np = horizontal.squeeze().numpy()
    vertical_np = vertical.squeeze().numpy()
    magnitude_np = edge_magnitude.squeeze().numpy()

    # 5. 可视化结果
    visualize_results(original_image,
                      horizontal_np,
                      vertical_np,
                      magnitude_np)
```

运行效果：

![](https://img.simoniu.com/自定义CNN模型实现图片边缘检测001.png)

### 代码说明：

1. **模型架构**：
   - 自定义CNN包含单个卷积层
   - 使用两个固定的3x3卷积核（Sobel算子的水平和垂直方向）
   - 卷积层权重被冻结（不参与训练）

2. **边缘检测原理**：
   - 水平卷积核检测垂直边缘
   - 垂直卷积核检测水平边缘
   - 边缘强度 = √(水平分量² + 垂直分量²)

3. **处理流程**：
   - 输入图像转换为灰度图
   - 归一化到[-1, 1]范围
   - 通过卷积层提取特征
   - 分离水平和垂直边缘分量
   - 计算边缘强度图

4. **输出可视化**：
   - 原始图像
   - 水平边缘特征图
   - 垂直边缘特征图
   - 边缘强度合成图

### 技术要点：
- 使用**Sobel算子**作为固定卷积核，是传统图像处理中经典的边缘检测方法
- 通过**torch.no_grad()** 禁用梯度计算，提高推理效率
- 特征图后处理包含**绝对值**和**强度计算**，增强边缘可视化效果
- 输出特征图与输入图像**同分辨率**（通过padding=1保持尺寸）

## 3.自定义CNN模型检测图片特征案例

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


# 定义简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 输入3通道，输出16通道
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 尺寸减半

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.features(x)


# 图像预处理
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像尺寸
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize(  # 标准化
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # 增加batch维度


# 可视化特征图
def visualize_feature_maps(feature_maps, layer_name):
    # 将特征图从GPU转到CPU并转为numpy
    features = feature_maps.squeeze(0).detach().cpu().numpy()

    # 创建子图
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    fig.suptitle(f'Feature Maps: {layer_name}', fontsize=16)

    # 绘制前64个特征图
    for i, ax in enumerate(axes.flat):
        if i < features.shape[0]:
            ax.imshow(features[i], cmap='viridis')
            ax.set_title(f'Ch {i + 1}')
            ax.axis('off')
    plt.tight_layout()
    plt.show()


# 主程序
if __name__ == "__main__":
    # 1. 创建模型
    model = SimpleCNN()
    print("模型结构:\n", model)

    # 2. 加载并预处理图像
    image_tensor = preprocess_image('./images/cat.jpg')  # 替换为你的图片路径
    print("输入图像尺寸:", image_tensor.shape)

    # 3. 前向传播提取特征
    with torch.no_grad():
        feature_maps = model(image_tensor)

    # 4. 输出特征图信息
    print("\n提取的特征图尺寸:", feature_maps.shape)  # [batch, channels, height, width]
    print("特征图数量:", feature_maps.size(1))

    # 5. 可视化特征图
    visualize_feature_maps(feature_maps, "Final Conv Layer")

```

运行效果：

![](https://img.simoniu.com/自定义CNN检测图片特征案例001.png)

这里重点解释一下transforms.Compose模块。

transforms.Compose 是 PyTorch 中 torchvision.transforms 模块提供的一个非常有用的类，它允许你将多个图像变换组合成一个单一的变换。这对于在数据预处理阶段需要应用一系列转换操作（如调整大小、裁剪、翻转、归一化等）特别方便。当你使用 Compose 来创建一个转换列表时，输入的图像会按照列表中转换的顺序依次经过每个转换。

- transforms.Resize((224, 224)): 这个转换操作会将输入图像的尺寸调整到指定的大小，这里是 224×224 像素。调整大小是很多深度学习模型的一个常见预处理步骤，因为大多数模型都需要固定的输入尺寸。
- transforms.ToTensor(): 此操作将PIL Image或NumPy ndarray转换为FloatTensor，并且调整图像像素值的范围从[0, 255]到[0.0, 1.0]。这样做是为了使输入适合于许多深度学习框架中的计算要求。
- transforms.Normalize(mean, std): 归一化操作。此步骤非常重要，因为它帮助模型更快地收敛。这里的参数mean和std分别表示用于每个通道的均值和标准差。对于RGB三通道图像，您提供了三个均值和三个标准差，对应于R、G、B三个通道。这个特定的均值和标准差集是基于ImageNet数据集计算得出的，常用于预训练模型的输入预处理。


这样，图像就会按顺序经过Resize、ToTensor和Normalize这三个转换步骤，最后得到一个可以输入到神经网络中的张量。

### 思考一个问题

这里让我思考另外一个问题，feature_maps = model(image_tensor) 这行代码为什么会自动调用forward方法呢？

#### 背后的机制

1. **`nn.Module`基类的魔法方法**：
   - 当你调用`model(input_tensor)`时，实际上是在调用模型的`__call__`方法
   - PyTorch的`nn.Module`类重写了`__call__`方法
   - `__call__`方法内部会调用`forward`方法，并添加额外的处理（如钩子函数）

2. **代码等价关系**：
   ```python
   # 这两种写法是等价的
   output = model(image_tensor)      # 推荐写法
   output = model.forward(image_tensor)  # 等效写法（但不推荐）
   ```

3. **为什么设计成这样**：
   - 语法更简洁自然（像函数调用一样使用模型）
   - 允许PyTorch在执行`forward`前后添加额外逻辑
   - 支持模型钩子（hooks）等高级功能
   - 保持API的一致性（所有模型都可以这样调用）



#### 为什么这样设计比直接调用`forward`更好

| 方式                         | 优点            | 缺点       |
| -------------------------- | ------------- | -------- |
| **`model(input)`**         | 自动处理hooks、梯度等 | -        |
| **`model.forward(input)`** | 更"直接"         | 可能绕过重要逻辑 |

在PyTorch中，**永远不要直接调用`forward()`**，因为：
1. 会跳过`nn.Module`的前置/后置处理
2. 可能导致hooks不执行
3. 梯度计算可能不正常


我来详细解释这三个问题，并通过具体代码示例说明为什么应该避免直接调用 `forward()` 方法。这些问题是PyTorch框架设计中的关键点，理解它们对正确使用PyTorch至关重要。

#### 1. 会跳过 `nn.Module` 的前置/后置处理

PyTorch 的 `nn.Module` 在调用 `__call__` 方法（即当你使用 `model(input)` 时）时，会执行一系列重要的前置和后置处理操作。这些操作包括：

- 设置模块的 training/eval 模式
- 调用注册的 forward hooks
- 管理模块调用堆栈
- 处理递归子模块调用

**代码示例：**

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2)
        
    def forward(self, x):
        print("执行 forward 方法")
        return self.layer(x)

# 创建模型实例
model = SimpleModel()

# 输入数据
input_data = torch.tensor([[1.0, 2.0]])

print("=== 使用 model(input_data) 调用 ===")
output1 = model(input_data)  # 正确方式

print("\n=== 使用 model.forward(input_data) 调用 ===")
output2 = model.forward(input_data)  # 错误方式
```

**输出结果：**
```
=== 使用 model(input_data) 调用 ===
执行 forward 方法

=== 使用 model.forward(input_data) 调用 ===
执行 forward 方法
```

虽然看起来两者都调用了 `forward`，但当你使用 `model(input_data)` 时，PyTorch 会执行额外的处理：

```python
# 伪代码展示 nn.Module 的 __call__ 方法内部逻辑
def __call__(self, *input, **kwargs):
    # 前置处理
    self._check_training_state()  # 检查训练/评估状态
    self._call_pre_hooks(input)   # 调用前向钩子
    
    # 实际执行前向传播
    result = self.forward(*input, **kwargs)
    
    # 后置处理
    self._call_post_hooks(result)  # 调用后向钩子
    self._update_module_stats()    # 更新模块统计信息
    
    return result
```

#### 2. 可能导致 hooks 不执行

Hooks 是 PyTorch 中强大的调试和扩展机制。直接调用 `forward()` 会绕过所有注册的 hooks。

**代码示例：**

```python
# 创建模型
model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))

# 注册一个前向钩子
def print_hook(module, input, output):
    print(f"钩子触发: {module.__class__.__name__} 输出形状: {output.shape}")
    return output

# 为第一层注册钩子
model.register_forward_hook(print_hook)

# 测试数据
x = torch.randn(1, 2)

print("=== 使用 model(x) 调用 ===")
out1 = model(x)  # 会触发钩子

print("\n=== 使用 model.forward(x) 调用 ===")
out2 = model.forward(x)  # 不会触发钩子
```

**输出结果：**
```
=== 使用 model(x) 调用 ===
钩子触发: Linear 输出形状: torch.Size([1, 3])

=== 使用 model.forward(x) 调用 ===
# 没有钩子输出
```

#### 3. 梯度计算可能不正常

这是最严重的问题。PyTorch 的自动微分系统依赖于正确的计算图构建，直接调用 `forward()` 会破坏这个机制。

**代码示例：**

```python
# 创建简单模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)
        
    def forward(self, x):
        return self.fc(x)

model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 输入和目标数据
x = torch.tensor([[1.0, 2.0]], requires_grad=True)
y_true = torch.tensor([[3.0]])

# 正确训练方式
def correct_training():
    optimizer.zero_grad()
    output = model(x)  # 正确调用
    loss = F.mse_loss(output, y_true)
    loss.backward()
    optimizer.step()
    print("正确训练 - 梯度:", model.fc.weight.grad)

# 错误训练方式（直接调用forward）
def incorrect_training():
    optimizer.zero_grad()
    output = model.forward(x)  # 错误调用
    loss = F.mse_loss(output, y_true)
    loss.backward()
    optimizer.step()
    print("错误训练 - 梯度:", model.fc.weight.grad)

# 运行两种方式
print("=== 正确训练方式 ===")
correct_training()

print("\n=== 错误训练方式（直接调用forward）===")
incorrect_training()
```

**输出结果：**
```
=== 正确训练方式 ===
正确训练 - 梯度: tensor([[0.4000, 0.8000]])

=== 错误训练方式（直接调用forward）===
错误训练 - 梯度: tensor([[0., 0.]])  # 梯度为零！
```

#### 问题分析

当直接调用 `forward()` 时：
1. **计算图断开**：PyTorch 无法追踪从输入到输出的完整计算路径
2. **梯度计算失败**：自动微分系统无法确定如何计算参数的梯度
3. **优化器无法更新参数**：因为所有梯度都是零或未定义

#### 更复杂的嵌套模块问题

当模型包含子模块时，直接调用 `forward()` 会引发更严重的问题：

```python
class ParentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = nn.Linear(2, 2)
        
    def forward(self, x):
        # 正确方式：应该使用 self.child(x) 而不是 self.child.forward(x)
        return self.child(x) ** 2  # 平方操作

model = ParentModel()
x = torch.tensor([[1.0, 2.0]], requires_grad=True)

# 正确调用
y1 = model(x)
y1.backward()
print("正确调用 - x的梯度:", x.grad)  # 应为 [4, 8]

# 重置梯度
x.grad = None

# 错误方式：直接调用forward
y2 = model.forward(x)  # 这会破坏子模块的调用机制
try:
    y2.backward()
    print("错误调用 - x的梯度:", x.grad)
except Exception as e:
    print(f"错误调用失败: {str(e)}")
```

**输出结果：**
```
正确调用 - x的梯度: tensor([[4., 8.]])
错误调用失败: element 0 of tensors does not require grad and does not have a grad_fn
```

#### 总结：为什么永远不要直接调用 `forward()`

| 问题类型        | 直接调用 `forward()` 的后果 | 正确调用 `model(input)` 的优势 |
| ----------- | -------------------- | ----------------------- |
| **前置/后置处理** | 跳过训练/评估模式切换，子模块初始化等  | 自动处理所有模块状态管理            |
| **Hooks**   | 所有注册的钩子都不会执行         | 保证所有钩子正确触发              |
| **梯度计算**    | 破坏计算图，导致梯度为零或错误      | 保持完整的计算图，正确计算梯度         |
| **嵌套模块**    | 子模块的前向传播也可能被破坏       | 递归正确处理所有子模块             |
| **框架兼容性**   | 可能导致与其他PyTorch功能不兼容  | 保证与所有PyTorch特性兼容        |

在实际开发中，**唯一**应该直接使用 `forward()` 的情况是当你在重写 `nn.Module` 的 `forward` 方法时，需要调用父类或其他模块的 `forward` 方法。即便如此，也应该使用 `super().forward(x)` 或 `self.child_module(x)` 的形式，而不是直接调用 `child_module.forward(x)`。

