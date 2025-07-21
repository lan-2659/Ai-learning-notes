import torch

def test02():
    # 定义数据
    x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)
    y = torch.tensor([3, 5, 7, 9, 11], dtype=torch.float)

    # 定义模型参数 a 和 b，并初始化
    a = torch.tensor([1], dtype=torch.float, requires_grad=True)
    b = torch.tensor([1], dtype=torch.float, requires_grad=True)
    # 学习率
    lr = 0.01  # 修正为更小的学习率
    # 迭代轮次
    epochs = 100

    for epoch in range(epochs):
        # 前向传播：计算预测值 y_pred
        y_pred = a * x + b

        # 定义损失函数
        loss = ((y_pred - y) ** 2).mean()

        if a.grad is not None and b.grad is not None:
            a.grad.zero_()
            b.grad.zero_()

        # 反向传播：计算梯度
        loss.backward()

        # 梯度下降
        with torch.no_grad():
            a.data -= lr * a.grad
            b.data -= lr * b.grad

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    print(f'a: {a.item()}, b: {b.item()}')

if __name__ == '__main__':
    test02()