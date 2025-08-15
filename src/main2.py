import cv2
import numpy as np
from ultralytics import YOLO
import time
import sys

model = YOLO("runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture("C:/Users/Amir/Desktop/4032/Test1.mp4")

# بررسی صحت باز شدن فایل ویدیو
if not cap.isOpened():
    print(f"خطا: فایل ویدیو '{cap}' قابل باز کردن نیست.")
    sys.exit(1)

# بارگذاری فایل نواحی لاین‌ها
lane_polygons = np.load("Box For Test1/lane_polygons.npy", allow_pickle=True)


output_width, output_height = 1280, 720
font = cv2.FONT_HERSHEY_SIMPLEX
blue_color = (255, 0, 0)
text_color = (255, 255, 255)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (output_width, output_height))
    results = model(frame)[0]

    lane_counts = [0] * len(lane_polygons)

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        # فقط کلاس CAR
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        lane_index = -1
        for i, polygon in enumerate(lane_polygons):
            if cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), (cx, cy), False) >= 0:
                lane_index = i
                lane_counts[i] += 1
                break

        # رسم باکس و مرکز
        cv2.rectangle(frame, (x1, y1), (x2, y2), blue_color, 2)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

        label = f"CAR ({lane_index+1})" if lane_index != -1 else "CAR"
        cv2.putText(frame, label, (x1, y1 - 8), font, 0.6, blue_color, 2)

    # رسم نواحی لاین‌ها و شمارش
    for i, polygon in enumerate(lane_polygons):
        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, f"Lane {i+1}: {lane_counts[i]}",(pts[0][0][0], pts[0][0][1] - 10),font, 0.7, text_color, 2)

    cv2.imshow("YOLOv8 Car Detection", frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
