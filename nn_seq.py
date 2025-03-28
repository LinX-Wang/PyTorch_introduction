import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class LinX(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self.conv1 = Conv2d(3, 32, stride=1, kernel_size=5,padding=2)
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32, 32, stride=1, kernel_size=5, padding=2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, stride=1, kernel_size=5, padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten = Flatten()
        # self.linear1 = Linear(1024, 64)
        # self.linear2 = Linear(64, 10)

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

linx = LinX()

input = torch.ones((64,3,32,32)) # batch,通道数，大小
output = linx(input)
print(output.shape)

writer = SummaryWriter("seq")
writer.add_graph(linx, input)
writer.close()