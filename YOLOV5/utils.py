import os
import torch
import cv2
from torch.utils.data import Dataset

class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.img_files = sorted(os.listdir(img_dir))
        self.label_files = sorted(os.listdir(label_dir))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 加载图像
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # 转换为Tensor
        
        # 加载标签
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        bboxes = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                class_id, x_center, y_center, w, h = map(float, line.strip().split())
                bboxes.append([class_id, x_center, y_center, w, h])

        bboxes = torch.tensor(bboxes)
        return img, bboxes
