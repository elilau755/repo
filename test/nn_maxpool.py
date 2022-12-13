import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())  # 想要测试数据集，变成Tensor类型

dataloader = DataLoader(dataset, batch_size=64)

# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)  # 若出现“long”报错，可以将元组内设为32浮点数
#
# input = torch.reshape(input, (-1, 1, 5, 5))  # -1表示模糊形状， 自动计算batchsize
# print(input.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False) # True, will use ceil instead of floor to compute the output shape

    def forward(self, input):
        output = self.maxpool1(input)
        return output

tudui = Tudui()  # create a nn
# output = tudui(input)
# print(output)
writer = SummaryWriter("P19_maxpool")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("maxpool_input", imgs, step)
    output = tudui(imgs)  # input进入nn；经过pooling之后的channel是不变的，所以不需要reshape
    writer.add_images("maxpool_output", output, step)
    step += 1

writer.close()
