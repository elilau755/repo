import os

root_dir = "dataset/train"
ants_target_dir = "ants_image"
bees_target_dir = "bees_image"
img_path1 = os.listdir(os.path.join(root_dir, ants_target_dir))
img_path2 = os.listdir(os.path.join(root_dir, bees_target_dir))
ants_label = ants_target_dir.split('_')[0]
bees_label = bees_target_dir.split('_')[0]
ants_out_dir = "ants_label"
bees_out_dir = "bees_label"
for i in img_path1:
    file_name = i.split('.jpg')[0]
    with open(os.path.join(root_dir, ants_out_dir, "{}.txt".format(file_name)), 'w') as f:
        f.write(ants_label)
for j in img_path2:
    file_name = j.split('.jpg')[0]
    with open(os.path.join(root_dir, bees_out_dir,  "{}.txt".format(file_name)), 'w') as p:
        p.write(bees_label)