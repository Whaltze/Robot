import cv2
import time
import math
import numpy as np

def process_frame(frame):  
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 255])
    upper_white = np.array([180, 55, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
   # res = cv2.bitwise_and(frame, frame, mask=mask)
    edges = cv2.Canny(mask, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_contour = max(contours, key = cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        center_x = x + w
        frame_center =(frame.shape[1])//2

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        frame = cv2.line(frame, (frame_center, 0), (frame_center, frame.shape[0]), (0, 0, 255), 2)
        

    cv2.imshow('Line Following', frame)

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        process_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()