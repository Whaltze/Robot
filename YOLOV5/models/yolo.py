import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans 

def generate_anchors(dataset, num_anchors=9, img_size=640):
    """使用KMeans聚类算法生成自适应锚框."""
    # 收集所有bounding box的宽高比例 (w, h)
    wh = []
    for label in dataset:
        for bbox in label["boxes"]:
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]  # 计算宽和高
            wh.append([w, h])

    wh = np.array(wh)
    kmeans = KMeans(n_clusters=num_anchors, random_state=0).fit(wh)

    # 将聚类结果缩放至图片大小范围内
    anchors = kmeans.cluster_centers_ * img_size / max(wh.max(0))
    return anchors

class YOLOv5WithAnchors(nn.Module):
    def __init__(self, num_classes=80, anchors=None):
        super(YOLOv5WithAnchors, self).__init__()
        self.anchors = anchors if anchors is not None else self._generate_default_anchors()
        
        # Backbone and other layers omitted for brevity...
        
    def _generate_default_anchors(self):
        """生成默认的锚框（使用手工定义的大小）."""
        return torch.tensor([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                             [59, 119], [116, 90], [156, 198], [373, 326]])

    def forward(self, x):
        # 此处省略主干网络和特征金字塔部分的实现...
        
        # 动态锚框推理
        pred = self._apply_anchors_to_output(x)
        return pred

    def _apply_anchors_to_output(self, x):
        """动态应用锚框到网络输出."""
        grid_size = x.shape[-2:]  # 获取当前网络输出的网格大小
        stride = 640 // grid_size[0]  # 动态推理时的步长
        
        anchors = (self.anchors / stride).to(x.device)
        # 在输出上应用锚框，计算边界框
        return self._compute_bounding_boxes(x, anchors)

    def _compute_bounding_boxes(self, x, anchors):
        """计算YOLO输出的bounding boxes."""
        # 假设模型输出为(batch_size, num_anchors * (5 + num_classes), grid_h, grid_w)
        num_anchors = len(anchors)
        grid_h, grid_w = x.shape[-2], x.shape[-1]

        # 重塑输出为适合 YOLO 框架的形式，并分离预测的每个部分
        x = x.view(-1, num_anchors, 5 + self.num_classes, grid_h, grid_w).permute(0, 1, 3, 4, 2)

        # 解码中心点偏移和锚框宽高，应用 sigmoid 到中心偏移量
        pred_boxes = torch.zeros_like(x[..., :4])  # x, y, w, h
        pred_boxes[..., 0:2] = torch.sigmoid(x[..., 0:2])  # x, y 经过 sigmoid 以限制在网格范围内
        pred_boxes[..., 2:4] = torch.exp(x[..., 2:4]) * anchors  # w, h

        return pred_boxes
