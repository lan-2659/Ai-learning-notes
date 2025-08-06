import torch

# 生成范围 [0, 10) 的 2x3 随机整数张量
data1 = torch.arange(1, 5, 1)
data2 = torch.arange(1, 5, 1)

print(data1 ** data2)