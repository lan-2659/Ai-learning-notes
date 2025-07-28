> Resnet是一种深度神经网络架构，被广泛用于计算机视觉任务，特别是图像分类。它是由微软研究院的研究员于2015年提出的，是深度学习领域的重要里程碑之一。

## 1.ResNet

如果说你对深度学习略有了解，那你一定听过大名鼎鼎的ResNet，正所谓ResNet 一出，谁与争锋？现如今2022年，依旧作为各大CV任务的backbone，比如ResNet-50、ResNet-101等。ResNet是2015年的ImageNet大规模视觉识别竞赛（ImageNet Large Scale Visual Recognition Challenge, ILSVRC）中获得了图像分类和物体识别的冠军，是中国人何恺明、张祥雨、任少卿、孙剑在微软亚洲研究院（AI黄埔军校）的研究成果。

![](https://img.simoniu.com/ResNet简介01.png)

## 2.网络过深导致的问题

在深度学习发展过程中，随着网络深度的增加，会出现梯度消失或梯度爆炸的问题，导 致网络难以训练。即使通过归一化等方法解决了梯度问题，还会面临退化问题，即网络深度增加时，模型的训 练误差和测试误差反而增大。

![](https://img.simoniu.com/网络过深带来的问题01.png)

从上面两个图可以看出，在网络很深的时候（56层相比20层），模型效果却越来越差了（误差率越高），并不是网络越深越好。

当你使用深度神经网络进行训练时，网络层可以被看作是一系列的函数堆叠，每个函数代表一个网络层的操作，这里我们就记作。在反向传播过程中，梯度是通过链式法则逐层计算得出的。假设每个操作的梯度都小于1，因为多个小于1的数相乘可能会导致结果变得更小。在神经网络中，随着反向传播的逐层传递，梯度可能会逐渐变得非常小，甚至接近于零，这就是梯度消失问题。

而如果经过网络层操作后的输出值大于1，那么反向传播时梯度可能会相应地增大。这种情况下，梯度爆炸问题可能会出现。梯度爆炸问题指的是在深度神经网络中，梯度逐渐放大，导致底层网络的参数更新过大，甚至可能导致数值溢出。

## 3.残差结构

在ResNet提出之前，所有的神经网络都是通过卷积层和池化层的叠加组成的。所以，Resnet对后面计算机视觉的发展影响是巨大的。ResNet的核心思想：引入了残差块（ResidualBlock）的概念。传统的神经网络层是直接学习输入到输出的映射，而残差块则是学习输入与输出之间的残差映射。

ResNet让网络学习层间的差异（或残差），而不是直接拟合原始目标函数。具体来说，如果我们期望的映射为H(x)，我们让网络去拟合残差F(x) = H(x) - x，然后最后输出为F(x) + x。这种结构通过“跳过连接”（skip connections）或“快捷连接”（shortcut connections）实现，即直接将输入添加到输出上。

打个比方，可以将残差块想象成一条高速公路，主路是神经网络的正常运算路径，而捷径则是一条可以直接从输入端到达输出端的辅路。这样，即使主路因为某些原因（如梯度消失）无法有效传递信息，信息仍然可以通过捷径直接流通，保证了数据的完整性和网络的学习能力。

![](https://img.simoniu.com/ResNet简介02.png)

通过这种方式，即便添加更多的层次，网络依然能够至少保持不变的性能，因为这些额外的层可以学习成为恒等映射，保持现有的性能。而在实际应用中，这些额外的层学到的残差能够帮助提升网络的性能。

ResNet的结构使得神经网络可以构建得更深，有助于解决更复杂的问题，同时还保持了网络的可训练性。自从提出以来，ResNet已经成为了许多视觉相关任务和其他需要深度网络的应用的基础架构。ResNet的变种，如ResNet-50、ResNet-101和ResNet-152等，表示的是网络含有的层数。

## 4.ResNet分类

- ResNet-18：是 ResNet 家族中相对较浅的网络，由 4 个残差块组构成，每个残差块组包含不同数量的残差块。它的结构简单，计算量相对较小，适合计算资源有限或对模型复杂度要求不高的场景，如一些小型图像数据集的分类任务。它在一些对实时性要求较高的应用中，如移动设备上的图像识别，也有一定的应用。
- ResNet-34：同样由 4 个残差块组组成，但相比 ResNet-18，它在某些残差块组中包含更多的残差块，网络深度更深，因此能够学习到更复杂的特征表示。它在中等规模的图像数据集上表现良好，在一些对模型性能有一定要求但又不过分追求极致精度的任务中较为常用。
- ResNet-50：是一个比较常用的 ResNet 模型，在许多计算机视觉任务中都有广泛应用。它使用了瓶颈结构（Bottleneck）的残差块，这种结构通过先降维、再卷积、最后升维的方式，在减少计算量的同时保持了模型的表达能力。该模型在图像分类、目标检测、语义分割等任务中，都能作为性能不错的骨干网络，为后续的任务提供有效的特征提取。
- ResNet-101：比 ResNet-50 的网络层数更多，拥有更强大的特征提取能力。它适用于大规模图像数据集和复杂的计算机视觉任务，如在大型目标检测数据集中，能够更好地捕捉目标的细节特征，提升检测的准确性。由于其深度和复杂度，在处理高分辨率图像或需要精细特征表示的任务时表现出色。
- ResNet-152：是 ResNet 系列中深度较深的网络，具有极高的特征提取能力。但由于其深度很大，计算量和参数量也相应增加，训练和推理所需的时间和资源较多。它通常用于对精度要求极高的场景，如学术研究中的图像识别挑战、大规模图像搜索引擎的图像特征提取等。


## 5.ResNet实现Mnist手写识别实例

- 项目需求：对手写数字进行识别。
- 数据集：此项目数据集来自MNIST 数据集由美国国家标准与技术研究所（NIST）整理而成，包含手写数字的图像，主要用于数字识别的训练和测试。该数据集被分为两部分：训练集和测试集。训练集包含 60,000 张图像，用于模型的学习和训练；测试集包含 10,000 张图像，用于评估训练好的模型在未见过的数据上的性能。
- 图像格式：数据集中的图像是灰度图像，即每个像素只有一个值表示其亮度，取值范围通常为 0（黑色）到 255（白色）。
- 图像尺寸：每张图像的尺寸为 28x28 像素，总共有 784 个像素点。
- 标签信息：每个图像都有一个对应的标签，标签是 0 到 9 之间的整数，表示图像中手写数字的值。


```python
import torch
from torch import nn  # 导入神经网络模块
from torch.utils.data import DataLoader  # 数据包管理工具，打包数据
from torchvision import datasets  # 封装了很对与图像相关的模型，数据集
from torchvision.transforms import ToTensor  # 数据转换，张量，将其他类型的数据转换成tensor张量
import torch.nn.functional as F  # 用于应用 ReLU 激活函数

'''下载训练数据集(包含训练集图片+标签)'''
training_data = datasets.MNIST(  # 跳转到函数的内部源代码，pycharm 按下ctrl+鼠标点击
    root='dataset',  # 表示下载的手写数字 到哪个路径。60000
    train=True,  # 读取下载后的数据中的数据集
    download=True,  # 如果你之前已经下载过了，就不用再下载了
    transform=ToTensor(),  # 张量，图片是不能直接传入神经网络模型
    # 对于pytorch库能够识别的数据一般是tensor张量
)

'''下载测试数据集（包含训练图片+标签）'''
test_data = datasets.MNIST(
    root='dataset',
    train=False,
    download=True,
    transform=ToTensor(),  # Tensor是在深度学习中提出并广泛应用的数据类型，它与深度学习框架（如pytorch，TensorFlow）
)  # numpy数组只能在cpu上运行。Tensor可以在GPU上运行，这在深度学习应用中可以显著提高计算速度。
print(len(training_data))
print(len(test_data))

# 设置每个批次的样本个数
train_dataloader = DataLoader(training_data, batch_size=64)  # 建议用2的指数当作一个包的数量
test_dataloader = DataLoader(test_data, batch_size=64)
'''判断是否支持GPU'''
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using {device} device')


# 定义残差块类，继承自 nn.Module
class ResBlock(nn.Module):
    def __init__(self, channels_in):
        # 调用父类的构造函数
        super().__init__()
        # 定义第一个卷积层，输入通道数为 channels_in，输出通道数为 30，卷积核大小为 5，填充为 2
        self.conv1 = torch.nn.Conv2d(channels_in, 30, 5, padding=2)
        # 定义第二个卷积层，输入通道数为 30，输出通道数为 channels_in，卷积核大小为 3，填充为 1
        self.conv2 = torch.nn.Conv2d(30, channels_in, 3, padding=1)

    def forward(self, x):
        # 输入数据通过第一个卷积层
        out = self.conv1(x)
        # 经过第一个卷积层的输出再通过第二个卷积层
        out = self.conv2(out)
        # 将输入 x 与卷积输出 out 相加，并通过 ReLU 激活函数
        return F.relu(out + x)


# 定义 ResNet 网络类，继承自 nn.Module
class ResNet(nn.Module):
    def __init__(self):
        # 调用父类的构造函数
        super().__init__()
        # 定义第一个卷积层，输入通道数为 1，输出通道数为 20，卷积核大小为 5
        self.conv1 = torch.nn.Conv2d(1, 20, 5)
        # 定义第二个卷积层，输入通道数为 20，输出通道数为 15，卷积核大小为 3
        self.conv2 = torch.nn.Conv2d(20, 15, 3)
        # 定义最大池化层，池化核大小为 2
        self.maxpool = torch.nn.MaxPool2d(2)
        # 定义第一个残差块，输入通道数为 20
        self.resblock1 = ResBlock(channels_in=20)
        # 定义第二个残差块，输入通道数为 15
        self.resblock2 = ResBlock(channels_in=15)
        # 定义全连接层，输入特征数为 375，输出特征数为 10
        self.full_c = torch.nn.Linear(375, 10)

    def forward(self, x):
        # 获取输入数据的批次大小
        size = x.shape[0]
        # 输入数据通过第一个卷积层，然后进行最大池化，最后通过 ReLU 激活函数
        x = F.relu(self.maxpool(self.conv1(x)))
        # 经过第一个卷积和池化的输出通过第一个残差块
        x = self.resblock1(x)
        # 经过第一个残差块的输出通过第二个卷积层，然后进行最大池化，最后通过 ReLU 激活函数
        x = F.relu(self.maxpool(self.conv2(x)))
        # 经过第二个卷积和池化的输出通过第二个残差块
        x = self.resblock2(x)
        # 将输出数据展平为一维向量
        x = x.view(size, -1)
        # 展平后的向量通过全连接层
        x = self.full_c(x)
        return x


# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    # 将模型设置为训练模式，这会影响一些层（如 Dropout、BatchNorm 等）的行为
    model.train()
    # 初始化批次编号
    batch_size_num = 1
    # 遍历数据加载器中的每个批次
    for x, y in dataloader:
        # 将输入数据和标签移动到指定设备（如 GPU）
        x, y = x.to(device), y.to(device)
        # 前向传播，计算模型的预测结果
        pred = model.forward(x)
        # 通过交叉熵损失函数计算预测结果与真实标签之间的损失值
        loss = loss_fn(pred, y)
        # 反向传播步骤：
        # 清零优化器中的梯度信息，防止梯度累积
        optimizer.zero_grad()
        # 反向传播计算每个参数的梯度
        loss.backward()
        # 根据计算得到的梯度更新模型的参数
        optimizer.step()
        # 从张量中提取损失值的标量
        loss_value = loss.item()
        # 每 100 个批次打印一次损失值
        if batch_size_num % 100 == 0:
            print(f'loss:{loss_value:7f}  [number:{batch_size_num}]')
        # 批次编号加 1
        batch_size_num += 1


# 定义测试函数
def test(dataloader, model, loss_fn):
    # 获取数据集的总样本数
    size = len(dataloader.dataset)
    # 获取数据加载器中的批次数量
    num_batches = len(dataloader)
    # 将模型设置为评估模式，这会影响一些层（如 Dropout、BatchNorm 等）的行为
    model.eval()
    # 初始化测试损失和正确预测的样本数
    test_loss, correct = 0, 0
    # 上下文管理器，关闭梯度计算，减少内存消耗
    with torch.no_grad():
        # 遍历数据加载器中的每个批次
        for x, y in dataloader:
            # 将输入数据和标签移动到指定设备（如 GPU）
            x, y = x.to(device), y.to(device)
            # 前向传播，计算模型的预测结果
            pred = model.forward(x)
            # 累加每个批次的损失值
            test_loss += loss_fn(pred, y).item()
            # 计算每个批次中预测正确的样本数并累加
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    # 计算平均测试损失
    test_loss /= num_batches
    # 计算平均准确率
    correct /= size
    # 打印测试结果
    print(f'Test result: \n Accuracy:{(100 * correct)}%,Avg loss:{test_loss}')


if __name__ == '__main__':
    model = ResNet().to(device)
    # 创建交叉熵损失函数对象
    loss_fn = nn.CrossEntropyLoss()
    # 创建 Adam 优化器，用于更新模型的参数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 定义训练的轮数
    epochs = 26
    # 开始训练循环
    for t in range(epochs):
        print(f'epoch{t + 1}\n--------------------')
        # 调用训练函数进行一轮训练
        train(train_dataloader, model, loss_fn, optimizer)
    print('Done!')
    # 调用测试函数进行测试
    test(test_dataloader, model, loss_fn)

```

运行效果：

```xml
60000
10000
Using cuda device
epoch1
--------------------
loss:0.076240  [number:100]
loss:0.165333  [number:200]
loss:0.072706  [number:300]
loss:0.242280  [number:400]
loss:0.129110  [number:500]
loss:0.132801  [number:600]
loss:0.108499  [number:700]
loss:0.054105  [number:800]
loss:0.128145  [number:900]
...
epoch26
--------------------
loss:0.000568  [number:100]
loss:0.000005  [number:200]
loss:0.000114  [number:300]
loss:0.000255  [number:400]
loss:0.002602  [number:500]
loss:0.086492  [number:600]
loss:0.000092  [number:700]
loss:0.000000  [number:800]
loss:0.011039  [number:900]
Done!
Test result: 
 Accuracy:99.02%,Avg loss:0.04639513847534029
```


