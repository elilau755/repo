from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("../logs") # 创建一个实例writer，对应事件存储到logs
image_path = r"D:\Python\pythonProject3\xiaotudui\dataset\train\ants_image\5650366_e22b7e1065.jpg"
img_PIL = Image.open(image_path) # 获取PIL类型
img_array = np.array(img_PIL) # 转成ndarray
print(type(img_array))
print(img_array.shape)

writer.add_image("train", img_array, 1, dataformats='HWC')
# 添加image add_image(self, tag, img_tensor, global_step) img_tensor要是ndarray型

for i in range(100):
    writer.add_scalar("y=2x", 2*i, i) # 添加标量
    # add_scalar(tag, axis_y， axis_x)

writer.close()