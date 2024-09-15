import cv2
import os
import torch


class DetectNumber:
	def __init__(self):
		# 载入模型
		self.model = torch.hub.load(f'{os.path.dirname(os.path.abspath(__file__))}', 'custom', path=f'{os.path.dirname(os.path.abspath(__file__))}/weights/best.pt', source='local', force_reload = True)
		self.model.conf = 0.5
	
	def detect(self, frame):
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		return self.model(frame_rgb).pandas().xyxy[0].to_numpy()
	
	def draw_detect(self, frame: cv2.Mat, result):
		for box in result:
			l, t, r, b = box[:4].astype('int')
			conf, num = box[4:6]
			# 绘制矩形框
			cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 5)
			# 绘制文字
			cv2.putText(frame, f"{num} {conf:.2f}", (l, t), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
		
		return frame


if __name__ == "__main__":
	cap = cv2.VideoCapture(0)
	model = DetectNumber()
	while cv2.waitKey(10) != ord('q'):
		ret, frame = cap.read()
		result = model.detect(frame)
		print(result)
		cv2.imshow("PPE", model.draw_detect(frame, result))
	
	cap.release()
	cv2.destroyAllWindows()
