import torch
import torch.nn as nn

class YOLO(nn.Module):
    def __init__(self):
        super(YOLO, self).__init__()
        # 简化的卷积神经网络
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 4096)
        self.fc2 = nn.Linear(4096, 7 * 7 * 5)  # 对应7x7网格和每个格子5个预测值

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        print(x.shape)  # 假设输出形状是 [batch_size, channels, height, width]

        # 展平张量
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, -1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        # 打印展平后的形状
        print(x.shape)
        # 确保目标形状匹配展平后的形状
        grid_size = int(x.size(1) ** 0.5)  # 计算网格大小（假设平方形网格）
        num_classes = 5  # 假设每个格子有 5 个值
        x = x.view(-1, grid_size, grid_size, num_classes)
        return x
