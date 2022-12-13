import torch
import torchvision
from torch import nn
from torch.nn import *
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=1)


class Tudui(nn.Module):   # 将nn_seq 写的神经网络复制过来
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
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

    def forward(self, x):  # 前向传播
        x = self.model1(x)
        return x

loss = CrossEntropyLoss()
tudui = Tudui()
optim = torch.optim.SGD(tudui.parameters(), lr=0.001) # 随机梯度下降 paramaters设置成tudui.parameters
for epoch in range(20):
    running_loss = 0.0  # 每一个epoch开始之前都将running_loss设置为0，最后观察每一轮总的loss差别
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        result_loss = loss(outputs, targets)
        optim.zero_grad()# First： 把网络模型中每一个可以调节参数的设置为0  从debug可以看出，后面一直在以下三行循环
        result_loss.backward()# 用optim优化器去对参数进行优化，优化器需要每个参数的梯度，故需要反向传播
        optim.step()  # 对每个参数进行调优
        # print(result_loss)  # 可以看到一个epoch的话，优化后的损失并不明显
        running_loss += result_loss
    print(running_loss)

    # tensor(21504.5605, grad_fn= < AddBackward0 >)
    # tensor(17893.9023, grad_fn= < AddBackward0 >)
    # tensor(16328.3623, grad_fn= < AddBackward0 >)
    # tensor(15396.0537, grad_fn= < AddBackward0 >)
    # tensor(14565.0742, grad_fn= < AddBackward0 >)
    # tensor(13742.0684, grad_fn= < AddBackward0 >)
    # tensor(12928.3506, grad_fn= < AddBackward0 >)
    # tensor(12130.0068, grad_fn= < AddBackward0 >)
    # tensor(11354.6445, grad_fn= < AddBackward0 >)
    # tensor(10590.4502, grad_fn= < AddBackward0 >)
    # tensor(9823.1895, grad_fn= < AddBackward0 >)
    # tensor(9031.5381, grad_fn= < AddBackward0 >)
    # tensor(8234.8418, grad_fn= < AddBackward0 >)
    # tensor(7386.8174, grad_fn= < AddBackward0 >)
    # tensor(6507.4790, grad_fn= < AddBackward0 >)
    # tensor(5636.9805, grad_fn= < AddBackward0 >)
    # tensor(4888.9849, grad_fn= < AddBackward0 >)
    # tensor(4508.8892, grad_fn= < AddBackward0 >)
    # tensor(4373.5327, grad_fn= < AddBackward0 >)
    # tensor(3869.9038, grad_fn= < AddBackward0 >)