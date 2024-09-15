import cv2
import os
import torch
import numpy as np

class DetectNumber:
    def __init__(self, model_path='weights/best.pt', confidence_threshold=0.5, device='cuda'):
        # 载入模型，指定设备（CUDA或CPU）
        try:
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
            self.model = torch.hub.load(
                os.path.dirname(os.path.abspath(__file__)), 
                'custom', 
                path=os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path),
                source='local', 
                force_reload=True
            ).to(self.device)
            self.model.conf = confidence_threshold  # 置信度阈值
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def detect(self, frame):
        # 转换为RGB格式，并确保输入的张量在模型运行的设备上
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb)
        
        # 将结果转换为NumPy数组进行后续处理
        return results.pandas().xyxy[0].to_numpy()

    def draw_detect(self, frame: cv2.Mat, result):
        for box in result:
            l, t, r, b = box[:4].astype(int)
            conf, num = box[4:6]
            
            # 绘制矩形框
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            
            # 绘制文字，包括检测的数字和置信度
            cv2.putText(frame, f"{int(num)} {conf:.2f}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        return frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # 打开摄像头
    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()

    model = DetectNumber()  # 初始化检测模型

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # 检测数字
        result = model.detect(frame)
        
        # 打印检测结果
        print(result)

        # 绘制检测结果并显示
        frame_with_detections = model.draw_detect(frame, result)
        cv2.imshow("Number Detection", frame_with_detections)

        # 按'q'键退出
        if cv2.waitKey(10) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
