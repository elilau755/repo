import torch
import torchvision
from torch import nn
from torch.nn import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2)) # -1自动算batchsize
print(input.shape)

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, 64)

class Tudui(nn.Module):   # create a nn model
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1 = ReLU(inplace=False)  # torch.nn.ReLU(inplace=False) 是否将input 替换成relu后的结果
                                                 # 一般情况下，使用False避免原始数据丢失
        self.sigmoid1 = Sigmoid()
    def forward(self, input):
        # output = self.relu1(input)
        output = self.sigmoid1(input)
        return output

tudui = Tudui()
# output = tudui(input)
# print(output)
writer = SummaryWriter("P20_sigmoid")
step = 0
for data in dataloader:   # CIFAR10 def __getitem__(self, index: int) -> Tuple[Any, Any]:   return img, target
    imgs, target = data
    writer.add_images("sigmoid_input", imgs, step)
    output = tudui(imgs)
    writer.add_images("sigmoid_output", output, step)
    step += 1

writer.close()

