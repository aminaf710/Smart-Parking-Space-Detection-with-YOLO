# Smart Parking Space Detection with YOLO
A real-time parking space detection system using YOLO and OpenCV.

---

![Parking Demo](assets/parking_output.gif)

## 📂 Project Structure

* `parking_setup.py` → Tool to manually define parking slots and main parking area
* `parking_detection.py` → Main script to detect cars and check parking availability
* `parking1.webcam` → Input video (replace with your own video)
* `parking7_output.mp4` → Output video with detected occupied/empty slots
* `README.md` → Project documentation

---

## ⚙️ Installation

1. Install required packages:

```
pip install opencv-python ultralytics numpy
```

2. (Optional) For faster detection, use a CUDA-enabled GPU.

---

## ▶️ Usage

1. **Define Parking Slots:**

   * Run `parking_setup.py`
   * Use left-click to add points for polygons
   * Right-click to finish a polygon
   * Press 'm' to toggle main parking area drawing
   * Press 'r' to reset
   * Press 'q' to finish and save coordinates

2. **Detect Cars & Parking Availability:**

   * Update the `parking_slots` and `main_parking_area` in `parking_detection.py` with coordinates from `parking_setup.py`
   * Run `parking_detection.py`
   * The video will show occupied spots in red and empty spots in green
   * Press 'q' to exit

---

## 🧠 How It Works

1. `parking_setup.py` lets the user define parking areas as polygons.
2. `parking_detection.py` applies YOLO to detect cars inside the main parking area.
3. Each parking slot is checked:

   * If a car’s center is inside the polygon → occupied
   * Else → empty
4. The system draws a real-time panel showing total, occupied, and empty spots.

---

## 📌 Notes

* Use clear and well-lit video for better detection.
* Make sure `YOLO` model path is correct.
* You can adjust the circle size and colors in the script for better visibility.
* Works with both video files and webcam input.
