import torch
from torch import nn
from torch.nn import *
from torch.utils.tensorboard import SummaryWriter


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # self.conv1 = Conv2d(3, 32, 5, padding=2) # CIFAR-20的结构 去网上随便找 输入3通道的32*32 kernel为5*5, stride和padding自己算
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32, 32, 5, padding=2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, 5, padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten = Flatten()
        # self.linear1 = Linear(1024, 64)
        # self.linear2 = Linear(64, 10)

        self.model1 = Sequential(    # 在sequential里面写网络结构,不同层之间用逗号分隔，这样即可把之前繁琐的self.conv1...删掉
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.model1(x)   # 这样一来就可以直接利用self.model1 = Sequential(),防止漏写
        return x

tudui = Tudui()
print(tudui)
input = torch.ones((64, 3, 32, 32))  # 全1
output = tudui(input)
print(output.shape)

writer = SummaryWriter("P22_CIFARseq")
writer.add_graph(tudui, input) # 计算图
writer.close()