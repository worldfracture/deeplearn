import torch
from torch import nn

# 定义一个网络模型
class MyLeNet5(nn.Module):
    # 初始化网络
    def __init__(self):
        super(MyLeNet5, self).__init__()

        self.c1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.Sigmoid = nn.Sigmoid()
        self.s2 = nn.AvgPool2d(kernel_size=2)
        self.c3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.s4 = nn.AvgPool2d(kernel_size=2)#, stride=2)
        self.f5 = nn.Linear(32*7*7,400)
        self.ReLU = nn.ReLU()

        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(400, 80)
        self.output = nn.Linear(80, 26)


    def forward(self, x):
        x = self.Sigmoid(self.c1(x))
        x = self.s2(x)
        x = self.Sigmoid(self.c3(x))
        x = self.s4(x)
        x = self.flatten(x)
        x = self.ReLU(self.f5(x))

        x = self.ReLU(self.f6(x))
        x = self.output(x)
        return x


