
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
cap = cv2.VideoCapture("Test1.mp4")

# Boxes
lane_polygons = np.load("Box For Test1/lane_polygons.npy", allow_pickle=True)

output_width, output_height = 1280, 720
font = cv2.FONT_HERSHEY_SIMPLEX

# Saving
frames_for_output = []
logs = []

# Color selection
def get_lane_color(count):
    if count < 5:
        return (0, 255, 0, 15)    # Green
    elif count < 9:
        return (0, 255, 255, 15)  # Yellow
    elif count < 14:
        return (0, 0, 255, 15)    # Red
    else:
        return (0, 0, 80, 15)     # Dark-Red

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (output_width, output_height))
    results = model(frame)[0]

    lane_counts = [0] * len(lane_polygons)

    # Count Cars
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        for i, polygon in enumerate(lane_polygons):
            if cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), (cx, cy), False) >= 0:
                lane_counts[i] += 1
                break

    # Painting Boxes
    overlay = frame.copy()
    for i, polygon in enumerate(lane_polygons):
        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
        b, g, r, alpha = get_lane_color(lane_counts[i])
        color = (b, g, r)

        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, alpha / 255.0, frame, 1 - alpha / 255.0, 0, frame)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        cv2.putText(frame, f"{lane_counts[i]}", (pts[0][0][0], pts[0][0][1]  - 10), font, 0.7, (0, 0, 0), 2)

    # Save Video
    frames_for_output.append(frame.copy())

    # Save log
    logs.append(",".join([str(c) for c in lane_counts]))

    cv2.imshow("Traffic-Detection", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()


save = input(" Do you want to Save Video and logs?(y/n) ").strip().lower()
if save == 'y':
    path = input(" Please Enter Save address:(EX: C:/Users/....) ").strip()

    # Saving-video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = path + "output_video.mp4"
    out = cv2.VideoWriter(out_path, fourcc, 30.0, (output_width, output_height))
    for f in frames_for_output:
        out.write(f)
    out.release()

    # Saving-log
    log_path = path + "traffic_log.txt"
    with open(log_path, 'w') as f:
        for idx, line in enumerate(logs):
            f.write(f"Frame {idx+1}: {line}\n")

    print("✅ Done")
else:
    print("❌ Video and logs not saved")

