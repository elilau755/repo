import torchvision
from torch.utils.tensorboard import SummaryWriter

# pytorch进行使用的时候，需要将数据集转成tensor的类型
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="../dataset", train=True, transform=dataset_transform, download=True) # download dataset
test_set = torchvision.datasets.CIFAR10(root="../dataset", train=True, transform=dataset_transform, download=True)

# print(test_set[0])
# print(test_set.classes)
#
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()
#
# print(test_set[0])

writer = SummaryWriter("p10") # 将日志文件保存到p10
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i) # 之间已经转成tensor了

writer.close()