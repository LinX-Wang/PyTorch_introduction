import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

inputs = torch.tensor([1,2,3], dtype=torch.float32)
targets = torch.tensor([1,2,5], dtype=torch.float32)

loss1 = L1Loss(reduction="sum")
loss2 = MSELoss()
result1 = loss1(inputs, targets)
result2 = loss2(inputs, targets)
print(result1)
print(result2)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3)) # 1 batch,3类

loss_cross = CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)