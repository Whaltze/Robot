import cv2
import numpy as np
import torch
import time

class PPE_detector:
    def __init__(self):    # 构造函数
        # 加载模型
        self.model = torch.hub.load('', 'custom', path='weights/yolov5s.pt', source='local')
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

            # 推理前，需要将画面转为RGB格式
            frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            # 执行推理过程
            results = self.model(frame_rgb)
            results_np = results.pandas().xyxy[0].to_numpy()

            print(results_np)
            
            # 绘制边界框
            for box in results_np:
                l,t,r,b = box[:4].astype('int')
                num = box[5]+1
                # 绘制矩形框
                cv2.rectangle(frame,(l,t),(r,b),(0,255,0),5)
                # 绘制文字
                cv2.putText(frame, str(num), (l,t), cv2.FONT_HERSHEY_PLAIN, 2.0,(0,0,255),2)

            # 显示画面
            cv2.imshow('PPE', frame)
            if cv2.waitKey(10)==ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
ppe = PPE_detector()
ppe.detect()
