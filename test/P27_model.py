import torch
from torch import nn

# 搭建神经网络   CIFAR-10其中有10个类别，故搭建的网络应该是一个10分类的网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()  # 父类初始化
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.model(x)
        return x

# 输入main 然后点击自动提示，在下面测试网络的正确性，给定一个输入shape，看输出shape是不是想要的10分类
if __name__ == '__main__':
    tudui = Tudui()
    input = torch.ones((64, 3, 32, 32))  # batch_size=64,64张图输入
    output = tudui(input)
    print(output.shape)