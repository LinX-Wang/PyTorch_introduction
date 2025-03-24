import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

input = torch.tensor([[1,2,0,3,1],
                     [0,1,2,3,1],
                     [1,2,1,0,0],
                     [5,2,3,1,1],
                     [2,1,0,1,1]])

input = torch.reshape(input, (1, 5, 5))

class LinX(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, x):
        x = self.maxpool1(x)
        return x

linx = LinX()

writer = SummaryWriter("maxpool")
step = 0
for data in dataloader:
    imgs, targets = data
    output = linx(imgs)

    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step += 1

writer.close()