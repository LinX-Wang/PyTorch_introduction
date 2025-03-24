import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Linear, Flatten, MaxPool2d, Conv2d
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=1)

class LinX(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model1 = Sequential(
            Conv2d(3, 32, stride=1, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, stride=1, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, stride=1, kernel_size=5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
linx = LinX()
optim = torch.optim.SGD(linx.parameters(), lr=0.01) # SGD 随机梯度下降

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        output = linx(imgs)
        result_loss = loss(output, targets)
        optim.zero_grad()    # 参数梯度初始设置为0
        result_loss.backward()
        optim.step()
        running_loss = running_loss + result_loss
    print(running_loss)


