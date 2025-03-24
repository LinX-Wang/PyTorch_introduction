import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

class LinX(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear1 = Linear(196608, 10)

    def forward(self, x):
        x = self.linear1(x)
        return x

linx = LinX()

for data in dataloader:
    imgs, targets = data
    # output = torch.reshape(1,1,1,-1) # 图像特征展平（高度是1）,效果等同于下方flatten

    output = torch.flatten(imgs)
    output = linx(output)
