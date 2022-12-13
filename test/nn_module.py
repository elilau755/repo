import torch
from torch import nn


class Tudui(nn.Module):   # 自定义一个类“Tudui模型”，继承父类module
    def __init__(self):    # 初始化
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

tudui = Tudui()  # 用Tudui模型创建的一个神经网络tudui
x = torch.tensor(1.0)  # 将1.0转化成tensor类型
output = tudui(x) # 将x放入神经网络tudui中，输出output
print(output)

