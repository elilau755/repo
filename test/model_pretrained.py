import torchvision

# train_data = torchvision.datasets.ImageNet("../dataset", split='train', download=True,
#                                             transform=torchvision.transforms.ToTensor())  # split 选择要训练集  **kwargs则是将一个可变的关键字参数的字典传给函数实参
# ERROR:The dataset is no longer publicly accessible. You need to download the archives externally and place them in the root directory.
from torch.nn import *

vgg16_false = torchvision.models.vgg16(pretrained=False, progress=True)  # 显示一个加载进度
vgg16_true = torchvision.models.vgg16(pretrained=True, progress=True) # 经过ImageNet预训练好的模型

print(vgg16_true)   # (6): Linear(in_features=4096, out_features=1000, bias=True) 最后输出的是1000个类

# 再来看一下之前用的CIFAR10（把数据分为了10类）
train_data = torchvision.datasets.CIFAR10("../dataset", train=True, transform=torchvision.transforms.ToTensor(),download=True)

# 迁移学习 Transfer Learnings
# 如何在CIFAR10这个数据集去应用VGG-16这个network model
# 如何利用现有的网络去改动它的结构，这样的话就可以避免去写这个VGG-16
vgg16_true.classifier.add_module('add_linear', Linear(in_features=1000, out_features=10))  # 在VGG-16 的基础上添加一个module，线性层使之输入为上一层的out_features=1000，变成10
                                                           # .classifier 就是将新加的module放入classifier 至于为什么是1000,可以从11行以后的都注释掉，看print(vgg16_true)最后的结果
print(vgg16_true)

# 当然也可以用另一种方法，直接修改而不是add_module 这里使用 vgg16_false 来演
print(vgg16_false)
vgg16_false.classifier[6] = Linear(in_features=4096, out_features=10)  # 直接将classifier[6]改为 out_features 为10
print(vgg16_false)
