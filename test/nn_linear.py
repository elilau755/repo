import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64, drop_last=True)  # 线性层实质上是矩阵相乘，不舍弃在最后数据会报错，不满足矩阵运算法则

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = Linear(196608, 10)    # 由print(out.shape)可以看出 in feature =196608,out feature 可设置为10
                                             # 将[1, 1, 1, 196608] 变成 [1, 1, 1, 10]

    def forward(self, input):
        output = self.linear1(input)
        return output

tudui = Tudui()


step = 0
for data in dataloader:
    imgs, target = data
    print(imgs.shape)
    # output = torch.reshape(imgs, (1, 1, 1, -1)) # 自动计算imgs_W  这行代码等价于flatten
    output = torch.flatten(imgs)  #  等同于将上面注释的代码
    print(output.shape)
    output = tudui(output)
    step += 1
    print(output.shape)
