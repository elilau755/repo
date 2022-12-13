#   CUDA 网络模型，损失函数，数据（输入，标注）可用，数据就是从data中取imgs，targets   Line(46)、（50）、（73）、（74）、（97）、（98）
import time

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

# from P27_model import * # P27_model.py需和P27_train.py在同一级路径下, 必须将test文件夹右键，标记目录为source root
# 准备数据集
from torch import nn
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10("../dataset", train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())

test_data = torchvision.datasets.CIFAR10("../dataset", train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据的长度为：{}".format(train_data_size)) # {}.format(x) 字符串格式化 将{}替换成x
print("测试数据的长度为：{}".format(test_data_size))

# 利用 DataLoader来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 网络模型创建
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
tudui = Tudui()
if torch.cuda.is_available():
    tudui = tudui.cuda() # 网络模型转移到cuda上利用gpu加速

# 损失函数  分类问题可以用交叉熵
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 1e-2     # hyper parameters  1-e3 = 1 * (10)^(-3) = 0.001
optimizer = torch.optim.SGD(params=tudui.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的次数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("../P30_traingpu")
start_time = time.time()
for i in range(epoch):   # 0~9
    print("第 {} 轮训练开始".format(i+1))  # i+1符合阅读习惯，因为i从0开始

    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad() # 优化前进行梯度清零
        loss.backward() # 调用loss的反向传播
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:   # 训练次数每100次，print loss
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数：{}， loss：{}".format(total_train_step, loss.item()))  # .item 将tensor-->真实值
            writer.add_scalar("train_loss", loss.item(), total_train_step)


    # 如何知道训练模型训练好/是否达到预期；在每个epoch训练完之后在test_dataloader上测试，以test_dataloader上的loss，acc来评估模型是否训练达到预期
    # 在测试的时候就不需要进行调优，只需要在现有的模型上进行测试
    # 测试步骤开始         注意这里的测试步骤依旧是在for epoch in range(10) 循环内，所以是每个epoch的
    total_test_loss = 0  # 求整个test_dataloader的loss
    total_correct = 0 # 整体测试集的正确的个数
    with torch.no_grad():   # no_grad 保证不会对其进行调优
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets) # 比较outputs和targets真实值之间的误差
            total_test_loss += loss.item()
            correct = (outputs.argmax(1) == targets).sum() # 每一个epoch的正确的个数，利用的算法是test_argmax.py，详见
            total_correct += correct
    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的Acc：{}".format(total_correct / test_data_size))  # 整体测试集正确个数 / 测试集数据长度 = 整体测试集Acc

    writer.add_scalar("test_Loss", total_test_loss, total_test_step)  # test_loss 纵轴, test_step 横轴
    writer.add_scalar("test_accuracy", total_correct / test_data_size, total_test_step)
    total_test_step += 1

    # 保存每一轮训练的模型
    torch.save(tudui, "tudui_{}_gpu1.pth".format(i))
    # torch.save(tudui.state_dict(), "tudui_{}.pth".format(i))  官方推荐
    print("模型已保存")

writer.close()




