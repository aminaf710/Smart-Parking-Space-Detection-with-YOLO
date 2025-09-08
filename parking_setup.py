import cv2
import numpy as np

# -----------------------------
# Video input
# -----------------------------
cap = cv2.VideoCapture("carpark.mp4")

ret, frame = cap.read()
if not ret:
    print("❌ Error opening video")
    exit()

clone = frame.copy()
parking_slots = []        # List to store individual parking slot polygons
current_polygon = []      # Points for the polygon being drawn
main_area = []            # Overall parking area polygon
drawing_main = False      # Flag: Are we drawing the main parking area?

# -----------------------------
# Mouse callback to draw polygons
# -----------------------------
def draw_polygon(event, x, y, flags, param):
    global current_polygon, frame, parking_slots, main_area, drawing_main

    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_polygon) > 2:
            pts = np.array(current_polygon, np.int32).reshape((-1, 1, 2))
            color = (0, 255, 0) if not drawing_main else (0, 0, 255)  # Green = slot, Red = main area
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

            if drawing_main:
                main_area.extend(current_polygon)
                print(f"Main parking area defined: {current_polygon}")
            else:
                parking_slots.append(current_polygon.copy())
                # Calculate center to put slot number
                M = cv2.moments(pts)
                if M["m00"] != 0:
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    cv2.putText(frame, str(len(parking_slots)), (cx-10, cy+10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                print(f"New slot #{len(parking_slots)} added: {current_polygon}")

        current_polygon = []

# -----------------------------
# Setup window and instructions
# -----------------------------
cv2.namedWindow("Parking Setup")
cv2.setMouseCallback("Parking Setup", draw_polygon)

print("Instructions:")
print(" - Left click to add points")
print(" - Right click to finish polygon")
print(" - Press 'm' to toggle main parking area drawing")
print(" - Press 'r' to reset")
print(" - Press 'q' to quit")

while True:
    temp = frame.copy()

    # Draw polygon while in progress (blue)
    if len(current_polygon) > 1:
        pts = np.array(current_polygon, np.int32).reshape((-1, 1, 2))
        cv2.polylines(temp, [pts], isClosed=False, color=(255, 0, 0), thickness=2)

    cv2.imshow("Parking Setup", temp)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("m"):
        drawing_main = not drawing_main
        print("Drawing main area:", drawing_main)

    elif key == ord("r"):
        frame = clone.copy()
        parking_slots = []
        current_polygon = []
        main_area = []
        drawing_main = False
        print("Reset all")

    elif key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()

print("✅ All parking slots:", parking_slots)
print("✅ Main parking area:", main_area)
