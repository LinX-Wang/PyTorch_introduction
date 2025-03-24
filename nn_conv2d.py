import torch
import torchvision
from numpy.ma.core import reshape
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)

class LinX(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = Conv2d(3, 6, stride=1, kernel_size=3,padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

linx = LinX()

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    output = linx(imgs)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)
    # torch.Size([64, 6, 30, 30]) ->[xxx, 3 , 30, 30]
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)
    step += 1

writer.close()