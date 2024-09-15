import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.yolo import YOLOv5WithAnchors, generate_anchors
from utils import YOLODataset  # 假设utils.py中定义了YOLODataset

def train_yolov5():
    dataset = YOLODataset(img_dir='dataset/images', label_dir='dataset/labels')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 自适应锚框生成
    anchors = generate_anchors(dataset)
    print("Generated anchors:", anchors)

    # 使用自适应锚框初始化YOLOv5模型
    model = YOLOv5WithAnchors(num_classes=10, anchors=anchors)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()  # 可以根据需求换成YOLO损失函数

    for epoch in range(20):
        model.train()
        running_loss = 0.0
        for images, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/20], Loss: {running_loss/len(dataloader)}")

if __name__ == "__main__":
    train_yolov5()
