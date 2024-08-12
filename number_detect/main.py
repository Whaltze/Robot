import cv2
import numpy as np
import torch

class number_detector:
    def __init__(self):  # 构造函数
        # 加载模型
        self.model = torch.hub.load('', 'custom', path='weights/best.pt', source='local')
        self.model.conf = 0.4  # 置信度
        # 获取视频流
        self.cap = cv2.VideoCapture(0)
    
    # 识别
    def detect(self):
        while True:
            # 获取视频流每一帧
            ret, frame = self.cap.read()
            
            # 画面翻转
            frame = cv2.flip(frame, 1)
            
            # 获取图像尺寸
            h, w = frame.shape[:2]
            
            # 定义ROI区域：取中间部分
            roi_w, roi_h = int(w * 0.5), int(h * 0.5)  # 中心区域的宽度和高度（可以根据需要调整）
            x1, y1 = (w - roi_w) // 2, (h - roi_h) // 2  # 左上角坐标
            x2, y2 = x1 + roi_w, y1 + roi_h  # 右下角坐标
            
            # 裁剪ROI区域
            roi = frame[y1:y2, x1:x2]
            
            # 推理前，需要将画面转为RGB格式
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # 执行推理过程
            results = self.model(roi_rgb)
            results_np = results.pandas().xyxy[0].to_numpy()
            
            # 绘制边界框和置信度
            for box in results_np:
                l, t, r, b = box[:4].astype('int')
                confidence = box[4]  # 置信度
                num = box[5]  # 类别
                
                # 将ROI的检测框映射回原图像
                l += x1
                t += y1
                r += x1
                b += y1
                
                # 绘制矩形框
                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 5)
                
                # 绘制类别和置信度
                text = f"{str(num)} {confidence:.2f}"  # 显示类别和置信度
                cv2.putText(frame, text, (l, t - 10), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
            
            # 显示画面
            cv2.imshow('PPE', frame)
            if cv2.waitKey(10) == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

# 实例化并启动检测
ppe = number_detector()
ppe.detect()

