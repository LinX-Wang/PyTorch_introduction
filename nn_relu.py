import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))  # -1为占位符，表示该维度可以根据其他维度自行判断得到

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class LinX(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, x):
        x = self.sigmoid1(x)
        return x

linx = LinX()

writer = SummaryWriter("sigmoid")
step = 0
for data in dataloader:
    imgs, targets = data
    output = linx(imgs)

    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step += 1

writer.close()