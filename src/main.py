import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture("C:/Users/Amir/Desktop/4032/Test1.mp4")

font = cv2.FONT_HERSHEY_SIMPLEX
output_width, output_height = 1280, 720

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (output_width, output_height))
    results = model(frame, verbose=False)[0]

    left_count = 0
    right_count = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id == 0 and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x_center = int((x1 + x2) / 2)

            # Left Or Right
            if x_center < output_width // 2:
                left_count += 1
            else:
                right_count += 1


    def get_color(count):
        if count < 19:
            return (0, 255, 0)  # Green
        elif 19 <= count <= 25:
            return (0, 255, 255)  # Yellow
        else:
            return (0, 0, 255)  # Red

    left_color = get_color(left_count)
    right_color = get_color(right_count)

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id == 0 and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x_center = int((x1 + x2) / 2)

            if x_center < output_width // 2:
                box_color = left_color
            else:
                box_color = right_color

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            # cv2.putText(frame, "CAR", (x1, y1 - 10), font, 0.6, box_color, 2)

    # Show
    cv2.putText(frame, f"Left Lane: {left_count}", (210, 55), font, 0.8, (0,0,0), 2)
    cv2.putText(frame, f"Right Lane: {right_count}", (850, 55), font, 0.8, (0,0,0), 2)

    # Center Line
    cv2.line(frame, (output_width // 2, 0), (output_width // 2, output_height), (255, 255, 255), 5)

    cv2.imshow("Traffic Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
