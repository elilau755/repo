import torch
import torch.nn.functional as F
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])  # 连续两个[[就是二维矩阵

kernel = torch.tensor([[1 ,2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])       # 卷积核

input = torch.reshape(input, (1, 1, 5, 5))  # (batch_size, channel, input_H, input_W)
kernel = torch.reshape(kernel, (1, 1, 3, 3)) # weight

output = F.conv2d(input, kernel, stride=1)
print(output)

output1 = F.conv2d(input, kernel, stride=2)
print(output1)

output2 = F.conv2d(input, kernel, stride=2, padding=1)
print(output2)

# torch.nn.Conv2d()
