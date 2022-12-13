from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# PYTHON的用法 ———》 tensor 数据类型
# 通过transforms.Tosensor看两个问题

# 2. 为什么我们需要Tensor数据类型


img_path = "dataset/train/ants_image/6240338_93729615ec.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

# 1. transforms该被如何使用（python）
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)  # img_tensor

# Normalize
print(tensor_img[0][0][0]) # print tensor_img 的CHW 第1 1 1
trans_norm = transforms.Normalize([5, 3, 1], [1, 2, 3])
img_norm = trans_norm(tensor_img)
print(img_norm[0][0][0])
writer.add_image("Norm_img", img_norm, 2)# 把最后的一个结果输出

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
# img PIL -> RESIZE -> img_resize PIL
img_resize = trans_resize(img) # PIL 类型
# img_resize PIL -> totensor -> img_reszie tensor
img_resize = tensor_trans(img_resize)
print(img_resize)
writer.add_image("Resize_img", img_resize, 0)

# Compose - resize -method 2   进行一个等比缩放，不改变高和宽的比例
trans_resize_2 = transforms.Resize(512) # PIL -> PIL
trans_compose = transforms.Compose([trans_resize_2, tensor_trans]) # compose()需要一个列表，transforms类型
# transforms.Compose([1, 2])    ！1, 2 的顺序要特别注意，后面是tensor  PIL -> tensor
img_resize_2 = trans_compose(img)
writer.add_image("Resize_img", img_resize_2, 1)

# RandomCrop
trans_random = transforms.RandomCrop((180, 400))  # 指定HW 进行裁剪
trans_compose_2 = transforms.Compose([trans_random, tensor_trans])
for i in range(10):  # 剪切10次
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCropHW", img_crop, i)


writer.close()
