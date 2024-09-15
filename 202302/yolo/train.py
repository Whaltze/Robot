import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import models
from models.yolo import YOLO
from utils import YOLODataset

def train():

    # 加载数据集
    dataset = YOLODataset(img_dir='data/images/', label_dir='data/labels/')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 初始化模型
    model = YOLO()
    criterion = torch.nn.MSELoss()  # 使用均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    for epoch in range(10):  # 简化为10个epoch
        running_loss = 0.0
        for images, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/10], Loss: {running_loss/len(dataloader)}')

if __name__ == "__main__":
    train()
