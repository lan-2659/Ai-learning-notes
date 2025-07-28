## 1.花分类数据集

花分类数据集一共有五个类别，分别是daisy（雏菊）、dandelion（蒲公英）、roses（玫瑰）、sunflowers（向日葵）和 tulips（郁金香）。

下面来看看这个数据集的目录结构，如下图所示：

![](https://img.simoniu.com/AlexNet花卉分类001.png)

## 2.AlexNet模型定义

model.py

```python
import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        # 通过初始化函数来定义网络在正向传播过程中所需要使用到的层结构
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 使用这个nn.Sequential能将一系列层结构打包，features用于提取图像特征
            # 不用每次写self.nn. 数据集比较小，卷积核的个数使用AlexNet中的一半
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            # padding一般只能传入int整型和tuple类型，比如tuple(1,2)代表上下各补一行0，左右各2列0；
            # 按照论文中精确padding，用nn.ZeroPad2d((1,2,1,2)),左1右2上1下2
            # pytorch卷积和池化过程中如果输出不是整数，会自动把多余数据舍弃掉
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27] 步长为1直接默认
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
        )
        # 包含全连接层  卷积层之后别忘了展平
        # 分开写是为了dropout只用于fc层
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # 以0.5的概率随机失活一些节点
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:  # 初始化权重
            self._initialize_weights()

    # 正向传播
    def forward(self, x):
        x = self.features(x)  # 进入卷积部分
        x = torch.flatten(x, start_dim=1)  # 展平处理，从第1维度开始展平，batch维度不动，也可以用view函数
        x = self.classifier(x)  # 进入全连接层
        return x

    # 初始化权重函数
    def _initialize_weights(self):
        for m in self.modules():  # 会遍历我们定义的每一个层结构
            if isinstance(m, nn.Conv2d):  # 如果层结构是卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 使用这个初始化变量方法来对我们卷积变量w进行初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 对偏置进行初始化
            elif isinstance(m, nn.Linear):  # 如果是全连接层
                nn.init.normal_(m.weight, 0, 0.01)  # 使用正态分布来初始化
                nn.init.constant_(m.bias, 0)  # 偏置初始化为0
```

## 3.模型训练

```python
import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from model import AlexNet

def main():
    # 使用GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train":
            transforms.Compose([
                transforms.RandomResizedCrop(224),  # 随机裁剪 224*224
                transforms.RandomHorizontalFlip(),  # 随机翻转 水平方向随机翻转进行数据增强
                transforms.ToTensor(),  # 转化为Tensor
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
        "val":
            transforms.Compose([
                transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    }

    #data_root = os.path.abspath(os.path.join(os.getcwd(),""))  # get data root path
    data_root = os.path.abspath(os.path.join(os.getcwd()))  # get data root path

    # os.getcwd()函数是获得这个文件所在的根目录，../表示返回上一层目录，../..就是返回上上层目录
    image_path = os.path.join(data_root, "data_set",
                              "flower_data")  # flower data set path
    # assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(
        image_path, "train"),
        transform=data_transform["train"])
    # 使用ImageFolder来加载数据集，train表示加载训练集数据，transform就是使用之前定义的数据预处理
    train_num = len(train_dataset)  # 训练集有多少张图片

    # 字典，类别：索引{'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx  # 去获取分类名称所对应的索引
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # 遍历flower_list这个字典，将key和value对调,也就是键值变成0，value变成daisy
    # trainset.classes可以直接获得类别的lise（待研究
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    # 通过json包将我们cla_dict这个字典进行编码，编码成json格式
    with open('class_indices.json',
              'w') as json_file:  # 保存class_indices.json文件，w是以写入的方式打开文件
        json_file.write(json_str)  # 能够在预测时方便读取它的信息

    batch_size = 32
 
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)
    # num_workers就代表加载数据集所使用的线程个数，win系统中不能设置为非0值，可以删去上一行直接写0

    validate_dataset = datasets.ImageFolder(root=os.path.join(
        image_path, "val"),
        transform=data_transform["val"])

    val_num = len(validate_dataset)  # 测试集的文件个数
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4,
                                                  shuffle=False,
                                                  num_workers=0)
    # 使用DataLoader去载入验证集
    print("using {} images for training, {} images for validation.".format(
        train_num, val_num))


    net = AlexNet(num_classes=5, init_weights=True)

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    # pata = list(net.parameters())  #这行代码是用来查看模型的参数
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 10
    save_path = './AlexNet.pth'  # 保存网络的路径
    best_acc = 0.0  # 定义这个参数是为了在后边训练网络中保存准确率最高的那次模型
    train_steps = len(train_loader)

    for epoch in range(epochs):
        # train
        net.train()  # 训练时使用net.train，验证时使用net.eval
        # 我们希望在训练的过程中进行Dropout，预测验证中不进行Dropout
        running_loss = 0.0
        # time_start = time.perf_counter()  #对训练一个epoch计时
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data  # 把数据集分为图像和对应的标签
            optimizer.zero_grad()  # 更新梯度信息
            outputs = net(images.to(device))  # 进行正向传播
            loss = loss_function(outputs, labels.to(device))  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            # print statistics
            running_loss += loss.item()  # loss累加

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(
                epoch + 1, epochs, loss)
        # print('%f s'%(time.perf_counter()-time_start))

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():  # 禁止pytorch在验证过程中对参数进行跟踪，验证中不会计算损失梯度
            val_bar = tqdm(validate_loader, file=sys.stdout)  # tqdm是为validate_loader创建一个带有进度条的可迭代对象，
            # 并将进度条输出到标准输出（sys.stdout)
            for val_data in val_bar:  # 遍历我们的验证集
                val_images, val_labels = val_data  # 把数据集分为图像和标签部分
                outputs = net(val_images.to(device))  # 将图片指认到设备上，传到网络得到输出
                predict_y = torch.max(outputs, dim=1)[1]  # 求得输出最大值，作为预测结果
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num  # 验证集的准确率
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:  # 如果验证集准确率大于历史最优准确率
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)  # 保存当前的权重

    print('Finished Training')


if __name__ == '__main__':
    main()
```

运行效果：

```xml
using cuda:0 device.
using 3670 images for training, 3670 images for validation.
train epoch[1/10] loss:1.399: 100%|██████████| 115/115 [00:23<00:00,  4.89it/s]
100%|██████████| 918/918 [00:13<00:00, 67.24it/s]
[epoch 1] train_loss: 1.342  val_accuracy: 0.481
...
train epoch[10/10] loss:0.357: 100%|██████████| 115/115 [00:23<00:00,  4.80it/s]
100%|██████████| 918/918 [00:13<00:00, 68.80it/s]
[epoch 10] train_loss: 0.806  val_accuracy: 0.762
Finished Training
```

## 4.模型预测

```python
import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import AlexNet
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(  # 依然是对数据先进行预处理
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img_path = "./images/daisy01.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)  # 直接使用PIL库载入一张图像

    plt.imshow(img)  # 简单展示一下这张图片
    # [N, C, H, W]
    img = data_transform(img)  # 对图片进行预处理
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)  # 扩充一个维度，添加一个batch维度

    # read class_indict
    json_path = './class_indices.json'  # 读取json文件，也就是索引对应的类别名称
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)  # 对json文件进行解码，解码成我们所需要的字典

    # create model
    model = AlexNet(num_classes=5).to(device)  # 初始化我们的网络

    # load model weights
    weights_path = "./AlexNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))  # 载入我们的网络模型

    model.eval()  # 进入eval模式，没有dropout的那个
    with torch.no_grad():  # 不跟踪变量的损失梯度
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()  # 将数据通过model进行正向传播得到输出
        # squeeze将输出进行压缩，把第一个维度的batch压缩掉了
        predict = torch.softmax(output, dim=0)  # softmax得到概率分布
        predict_cla = torch.argmax(predict).numpy()  # 概率最大处所对应的索引值

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    # 打印预测名称，已经对应类别的概率
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
```

运行效果：

![](https://img.simoniu.com/AlexNet花卉分类002.png)








