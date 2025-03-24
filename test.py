import torch
import torchvision
from PIL import Image
from torch import nn

# 加载图像
image_path = "airplane.png"
image = Image.open(image_path)
print(image)
image = image.convert('RGB')  # 将 PNG 图像转换为 RGB 格式

# 图像预处理
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])
image = transform(image)
print(image.shape)

# 定义模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型并移动到设备
model = torch.load("tudui_9.pth", map_location=device)
model = model.to(device)

# 调整输入张量形状并移动到设备
image = torch.reshape(image, (1, 3, 32, 32)).to(device)

# 推理
model.eval()
with torch.no_grad():
    output = model(image)

# 输出结果
print(output)
print(output.argmax(1))