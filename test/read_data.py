from torch.utils.data import Dataset
from PIL import Image
import os

class Mydata(Dataset):
    def __init__(self, root_dir, label_dir):  # 初始化
        self.root_dir = root_dir # self定义全局变量
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir) # join两个路径
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx] # 这时候只知道这个名称
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name) # 每一个图片的一个位置
        img = Image.open(img_item_path) # 读取图片
        label = self.label_dir
        return img, label

    def __len__(self):  # 数据集有多长
        return len(self.img_path)


root_dir = "dataset/train"
ants_label_dir = "ants" # ants label
bees_label_dir = "bees" # bees label
ants_dataset = Mydata(root_dir, ants_label_dir) # 含有__init__ 初始化的四个变量
bees_dataset = Mydata(root_dir, bees_label_dir)

train_dataset = ants_dataset + bees_dataset # 将两个label的数据集拼接
