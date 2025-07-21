import torch
from torch import nn
from torch import optim


# 定义数据
x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)
y = torch.tensor([3, 5, 7, 9, 11], dtype=torch.float)

model = nn.Linear(5, 1)
loss_fn = nn.MSELoss()
sgd = optim.SGD(model.parameters(), lr=0.01)

epochs = 100

for epoch in range(epochs):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    sgd.step()
    sgd.zero_grad()
    if epoch == 0 or (epoch + 1) % 10 == 0:
        print(f'Epoch[{epoch+1}/{epochs}] loss: {loss.item():.2f}')

print(list(model.parameters()))