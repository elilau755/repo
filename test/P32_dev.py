import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "../dataset/P32_dev_imgs/dog.png"
image = Image.open(image_path)
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                           torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()  # 父类初始化
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 之前P27train是用方式一保存的，故用方式一加载
model = torch.load("tudui_9_gpu2.pth", map_location=torch.device('cpu'))   # tudui_0,1,2... 越多轮，验证的越准确~！
# 使用GPU保存的，在加载时需要CPU加载的，在后面添加 “map_location=torch.device('cpu')”    映射到cpu上
# 使用CPU保存的，在加载时需要CPU加载的，则不需要添加
# 比如 model = torch.load("tudui_0_cpu_acc.pth") 即可
print(model)
# 调用torch.reshape 加上batch_size
image = torch.reshape(image, (1, 3, 32, 32))
# 最好加上，网络模型中有Dropout，BatchNorm，不加的话预测也许有问题
model.eval()
with torch.no_grad():
    output = model(image)
print(output) # Expected 4-dimensional input for 4-dimensional weight [32, 3, 5, 5], but got 3-dimensional input of size [3, 32, 32] instead,
              # 说明没加batch_size

print(output.argmax(1))

# tensor([[-1.0105, -4.2494,  1.3171,  2.5982,  0.9480,  3.0301,  0.1900,  1.4567,
#          -3.8113, -1.5727]])
# tensor([5])
# 可以看到在CIFAR10上第五个类别正好是狗，这里验证的图片分类到狗，所以验证正确

# 至于怎么查看类别代表的是什么，可以在P27_train.py Line(14) 打断点debugP27_train.py
# test_data = torchvision.datasets.CIFAR10("../dataset", train=False, download=True,
#                                          transform=torchvision.transforms.ToTensor())
# 点击 train_data --> class_to_idx 即可
# tensor([[airplane, automobile,  bird,  cat,  deer,  dog,  frog,  horse,
# #          ship, truck, __len__]])
