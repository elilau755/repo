import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False) # 模型是没有经过ImageNet预训练的
# 保存方式1  模型结构+模型参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2  模型参数（官方推荐）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")  # 将状态（网络模型参数）保存成dictionary字典格式（python中一个数据格式）
                                # 现在不保存网络模型，只保存参数
# 陷阱    比如现在自定义了一个网络结构
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)

    def forward(self):
        x = self.conv1(x)
        return x

tudui = Tudui()
torch.save(tudui, "tudui_method1.pth") # 保存方式1

