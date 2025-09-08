import cv2
from ultralytics import YOLO
import numpy as np

# ---------------------------
# 1. Load YOLO model
# ---------------------------
model = YOLO("yolov8n.pt")

# ---------------------------
# 2. Define parking slots manually (from parking_setup.py output)
# ---------------------------
parking_slots = [[(101, 326), (30, 325), (78, 263), (140, 266), (105, 328)], ...]  # Example
main_parking_area = [(4, 357), (76, 198), (564, 210), (637, 296), (633, 350), (8, 358)]

# ---------------------------
# 3. Open input video and setup output
# ---------------------------
cap = cv2.VideoCapture("parking1.webcam")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("parking7_output.mp4", fourcc, 30,
                      (int(cap.get(3)), int(cap.get(4))))

# Create mask for main parking area
ret, frame = cap.read()
if not ret:
    print("❌ Error opening video")
    exit()

mask = np.zeros_like(frame[:, :, 0])
pts = np.array(main_parking_area, np.int32).reshape((-1, 1, 2))
cv2.fillPoly(mask, [pts], 255)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ---------------------------
# 4. Process frames
# ---------------------------
while True:
    success, frame = cap.read()
    if not success:
        break

    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Detect cars only (YOLO class 2 = car)
    results = model(masked_frame, stream=True, classes=[2])
    cars = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cars.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

    total_parking = len(parking_slots)
    empty_slots = 0
    occupied_slots = 0
    empty_messages = []

    # Check each slot
    for idx, slot in enumerate(parking_slots):
        pts_slot = np.array(slot, np.int32).reshape((-1, 1, 2))
        M = cv2.moments(pts_slot)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = slot[0]

        color = (0, 255, 0)  # green = empty
        occupied = False
        for car in cars:
            car_cx = (car[0] + car[2]) // 2
            car_cy = (car[1] + car[3]) // 2
            if cv2.pointPolygonTest(pts_slot, (car_cx, car_cy), False) >= 0:
                color = (0, 0, 255)  # red = occupied
                occupied = True
                break

        if occupied:
            occupied_slots += 1
        else:
            empty_slots += 1
            empty_messages.append(f"Number {idx + 1} is now empty")

        cv2.circle(frame, (cx, cy), 15, color, -1)
        cv2.putText(frame, str(idx + 1), (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw panel
    panel_x, panel_y = 10, 30
    cv2.rectangle(frame, (5,5), (300, 60 + 20*len(empty_messages)), (50,50,50), -1)
    cv2.putText(frame, f"Total Parking: {total_parking}", (panel_x, panel_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"Occupied: {occupied_slots}", (panel_x, panel_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    cv2.putText(frame, f"Empty: {empty_slots}", (panel_x, panel_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    for i, msg in enumerate(empty_messages):
        cv2.putText(frame, msg, (panel_x, panel_y + 60 + 20*i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("Parking Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release
cap.release()
out.release()
cv2.destroyAllWindows()
