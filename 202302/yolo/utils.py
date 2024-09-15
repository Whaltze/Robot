import os
import torch
import cv2
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.img_files = sorted(os.listdir(img_dir))
        self.label_files = sorted(os.listdir(label_dir))
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # 转换为 torch.Tensor
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 加载图像
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = Image.open(img_path).convert("RGB")  # 使用 PIL 加载图像
        img = self.transform(img)  # 使用 transform 转换为 tensor
        
        # 加载标签
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        bboxes = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                class_id, x_center, y_center, w, h = map(float, line.strip().split())
                bboxes.append([class_id, x_center, y_center, w, h])

        bboxes = torch.tensor(bboxes)  # 转换为 tensor
        return img, bboxes
