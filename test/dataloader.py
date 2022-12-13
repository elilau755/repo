import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)  # 加载测试集数据

# 测试数据集中第一张图片及target
img, target = test_data[0]
print(img.shape)
print(target)
# img, target = test_data[0]
# print(img)
# print(target)

# 取出test_loader中每一个返回 return img， target
writer = SummaryWriter("../dataloader")  # 日志保存路径名
for epoch in range(2):  # 对比两轮由shuffle buller不同返回的数据
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(target)
        writer.add_images("Epoch: {}".format(epoch), imgs, step)  # add_images()  Add batched image data to summary.
        step += 1

writer.close()
