import torchvision
from torch import nn
from torch.nn import *
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)


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
for data in dataloader:
    imgs, targets = data
    outputs = tudui(imgs)
    # print(outputs)
    # print(targets)
    result_loss = loss(outputs, targets)
    # print(result_loss)
    result_loss.backward()  # 反向传播利用的是loss  可以把断点打在这一行，通过debug观察，tudui->module->protect->0->weight->grad
                            # 可以通过下一节讲的优化器来对这些参数进行优化
    print("ok")