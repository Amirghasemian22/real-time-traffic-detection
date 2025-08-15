import cv2
import numpy as np

# تعداد کل لاین‌ها
NUM_LANES = 6

# لیست نهایی تمام پلیگان‌ها
lane_polygons = []

# ناحیه موقتی هر لاین
current_polygon = []

# شمارنده لاین
lane_index = 0

# اسم پنجره
window_name = "Draw Lane Areas - Lane #{}"

# ماوس کال‌بک برای ثبت نقاط
def mouse_callback(event, x, y, flags, param):
    global current_polygon, lane_index

    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))

# بارگذاری فریم نمونه از ویدیو
cap = cv2.VideoCapture("C:/Users/Amir/Desktop/4032/Test1.mp4")
ret, frame = cap.read()
cap.release()

# تغییر سایز برای راحتی
frame = cv2.resize(frame, (1280, 720))
clone = frame.copy()

cv2.namedWindow(window_name.format(lane_index + 1))
cv2.setMouseCallback(window_name.format(lane_index + 1), mouse_callback)

while lane_index < NUM_LANES:
    temp_frame = clone.copy()

    # رسم نقاط فعلی
    for point in current_polygon:
        cv2.circle(temp_frame, point, 5, (255, 0, 0), -1)

    # رسم خطوط بین نقاط
    if len(current_polygon) >= 2:
        cv2.polylines(temp_frame, [np.array(current_polygon)], False, (0, 255, 255), 1)

    cv2.putText(temp_frame, f"Lane #{lane_index+1}: Click points, press ENTER to save", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow(window_name.format(lane_index + 1), temp_frame)
    key = cv2.waitKey(1)

    if key == 13:  # ENTER key
        if len(current_polygon) >= 3:
            lane_polygons.append(np.array(current_polygon))
            current_polygon = []
            lane_index += 1
            if lane_index < NUM_LANES:
                cv2.destroyWindow(window_name.format(lane_index))
                cv2.namedWindow(window_name.format(lane_index + 1))
                cv2.setMouseCallback(window_name.format(lane_index + 1), mouse_callback)
        else:
            print("برای ثبت یک ناحیه، حداقل سه نقطه لازم است.")
    elif key == 27:  # ESC key
        break

cv2.destroyAllWindows()

# ذخیره کردن نواحی
np.save("lane_polygons.npy", np.array(lane_polygons, dtype=object))

print("✔️ نواحی 6 لاین با موفقیت ذخیره شدند.")
