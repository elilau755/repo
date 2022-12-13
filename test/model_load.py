import torch
from model_save import *
# 方式1 --> 保存方式1，加载模型
import torchvision
from torch import nn

model = torch.load("vgg16_method1.pth")
print(model)

# 方式2 --> 保存方式2
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth")) # 将保存的dict格式转换模型＋参数
# model2 = torch.load("vgg16_method2.pth")
# print(model2)  # print 是一种dict格式
print(vgg16)

# 陷阱(方式1）
# class Tudui(nn.Module):   # ！在model前加上模型，就可以避免这个陷阱，但是一般来说
#     def __init__(self):   # ！都是将model放在一个文件（如：model_save.py）中，通过from model_save import * 全部导入
#         super(Tudui, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, 3)
#
#     def forward(self):
#         x = self.conv1(x)
#         return x

model = torch.load("tudui_method1.pth") # 方式1 加载模型
print(model) # AttributeError: Can't get attribute 'Tudui' on <module '__main__'
# 模型无法访问这个自定义的方式