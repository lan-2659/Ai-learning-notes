# 基本组件认知

先初步认知，他们用法基本一样的，后续在学习深度神经网络和卷积神经网络的过程中会很自然的学到更多组件！

官方文档：https://pytorch.org/docs/stable/nn.html

## 线性层组件

`nn.Linear` 是 PyTorch 中的一个非常重要的模块，用于实现全连接层（也称为线性层）。



## 激活函数组件

激活函数的作用是在隐藏层引入非线性，使得神经网络能够学习和表示复杂的函数关系，使网络具备非线性能力，增强其表达能力。

常见激活函数：

**sigmoid函数：**

    import torch.nn.functional as F
    sigmoid = F.sigmoid()

**tanh函数：**

    import torch.nn.functional as F
    tanh = F.tanh

**ReLU函数：**

    import torch.nn as nn
    relu = nn.ReLU()

**LeakyReLU函数**：

    import torch.nn as nn
    leaky_relu = nn.LeakyReLU(negative_slope=0.01)

**softmax函数：**

    import torch.nn.functional as F
    softmax = F.softmax
## 损失函数组件

损失函数的主要作用是量化模型预测值（$\hat{y}$）与真实值（$y$）之间的差异。通常，损失函数的值越小，表示模型的预测越接近真实值。

PyTorch已内置多种损失函数，在构建神经网络时随用随取！

文档：https://pytorch.org/docs/stable/nn.html#loss-functions

根据任务类型（如回归、分类等），损失函数可以分为以下几类：

### 回归任务的损失函数：

1. **均方误差损失**

    import torch.nn as nn
    loss_fn = nn.MSELoss()

2. **L1 损失**

也叫做MAE（Mean Absolute Error，平均绝对误差）


```
import torch.nn as nn
loss_fn = nn.L1Loss()
```
### **分类任务的损失函数：**

1. **交叉熵损失**

```
cross_entropy_loss = nn.CrossEntropyLoss()
```

2. **二元交叉熵损失**

```
bce_loss = nn.BCELoss()
bce_with_logits_loss = nn.BCEWithLogitsLoss()
```
## 优化器

在PyTorch中，**优化器（Optimizer）是用于更新模型参数以最小化损失函数的核心工具。**

官方文档：https://pytorch.org/docs/stable/optim.html

PyTorch 在 torch.optim 模块中提供了多种优化器，常用的包括：

- **SGD（随机梯度下降）**
- **Adagrad（自适应梯度）**
- **RMSprop（均方根传播）**
- **Adam（自适应矩估计）**






**注意：上面的内容都只是组件初步认识**