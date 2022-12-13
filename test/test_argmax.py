import torch

outputs = torch.tensor([[0.1, 0.2],
                        [0.05, 0.4]])

print(outputs.argmax(axis=1))    # axis=1，水平轴 -->0-->1-->  argmax对应的水平方向上最大元素对应的是0还是1
print(outputs.argmax(axis=0))    # axis=0, 就是纵轴， 在纵轴方向上 0.1和0.4最大 故 0.1 对应是的 0 ； 0.4 对应的是 1

preds = outputs.argmax(axis=1)
targets = torch.tensor([0, 1])
print(preds == targets)
print((preds == targets).sum()) # 可以计算出相等的次数，从而应用到模型的准确率上