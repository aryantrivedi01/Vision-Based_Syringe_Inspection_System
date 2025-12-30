import cv2
import numpy as np
import math
import time
from picamera2 import Picamera2
from ultralytics import YOLO
import os

# TEMPLATE MATCHING CONFIG
template_folder = "/home/jayesh/pbl/template_matching/original_template"
threshold = 0.8

def classify_with_templates(image, template_folder, threshold=0.85):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for filename in os.listdir(template_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            template_path = os.path.join(template_folder, filename)
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is None or gray_image.shape[0] < template.shape[0] or gray_image.shape[1] < template.shape[1]:
                continue
            result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val >= threshold:
                return "PASS"
    return "FAIL"

def detect_bevel_angle_from_frame(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    h, w = edges.shape
    ys, xs = np.where(edges > 0)
    if len(ys) == 0:
        return None
    tip_y = ys.min()

    band_h = int(h * 0.3)
    y1 = tip_y
    y2 = min(h, tip_y + band_h)
    band = edges[y1:y2, :]

    lines = cv2.HoughLinesP(
        band,
        rho=1,
        theta=np.pi/180,
        threshold=20,
        minLineLength=max(10, w // 10),
        maxLineGap=4
    )
    if lines is None:
        return None

    deviations = []
    for x1, y1b, x2, y2b in lines[:, 0]:
        y1_img = y1b + tip_y
        y2_img = y2b + tip_y
        dx = x2 - x1
        dy = y2_img - y1_img
        if abs(dy) < 2:
            continue
        angle = abs(math.degrees(math.atan2(dy, dx)))
        dev = abs(90 - angle)
        if 3 <= dev <= 25:
            deviations.append(dev)

    if not deviations:
        return None

    bevel_angle = np.median(deviations)
    return bevel_angle

def draw_button(frame, rect, label):
    x1, y1, x2, y2 = rect
    cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 150, 255), -1)
    cv2.putText(frame, label, (x1 + 35, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

# Init camera and model
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (640, 480)}))
picam2.set_controls({"AfMode": 2})
picam2.start()

model = YOLO("/home/jayesh/pbl/yolo lite/runs/detect/train2/weights/best.pt")
refresh_rect = (250, 490, 390, 530)

# State
taper_text = "Taper: N/A"
uneven_text = "Unevenness: N/A"
yolo_text = "Detection: N/A"
result_text = "Result: N/A"
button_pressed = False

last_bbox = None
last_label = ""

def on_mouse(event, x, y, flags, param):
    global button_pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        if refresh_rect[0] <= x <= refresh_rect[2] and refresh_rect[1] <= y <= refresh_rect[3]:
            button_pressed = True

cv2.namedWindow("Live Feed")
cv2.setMouseCallback("Live Feed", on_mouse)

try:
    while True:
        frame = picam2.capture_array()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        canvas = np.zeros((540, 640, 3), dtype=np.uint8)
        canvas[:480] = frame[:, :, :3]

        if button_pressed:
            yolo_text = "Detection: Not Found"
            taper_text = "Taper: N/A"
            uneven_text = "Unevenness: N/A"
            result_text = "Result: FAIL"

            results = model(frame_rgb)[0]
            if results.boxes:
                box = results.boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                
                label = "good" if cls_id == 0 else "defect"
                color = (0, 255, 0) if label == "good" else (0, 0, 255)

                # Store for later persistent drawing
                last_bbox = (x1, y1, x2, y2)
                last_label = label

                yolo_text = f"Detection: {label.upper()}"

                # TAPER ANGLE CALCULATION
                pad = 15
                x1p = max(0, x1 - pad)
                y1p = max(0, y1 - pad)
                x2p = min(frame.shape[1], x2 + pad)
                y2p = min(frame.shape[0], y2 + pad)

                readings = []
                for _ in range(7):
                    f2 = picam2.capture_array()
                    roi2 = f2[y1p:y2p, x1p:x2p]
                    a = detect_bevel_angle_from_frame(roi2)
                    if a is not None:
                        readings.append(a)
                    time.sleep(0.04)

                if readings:
                    angle = float(np.median(readings))
                    taper_text = f"Taper: {angle:.2f}"
                else:
                    taper_text = "Taper: N/A"

                une = classify_with_templates(frame, template_folder, threshold)
                uneven_text = f"Unevenness: {une}"

                if une == "PASS":
                    result_text = "Result: PASS"

            button_pressed = False

        # Draw stored YOLO detection
        if last_bbox:
            x1, y1, x2, y2 = last_bbox
            color = (0, 255, 0) if last_label == "good" else (0, 0, 255)
            label_text = f"{last_label}"
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(canvas, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(canvas, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        draw_button(canvas, refresh_rect, "REFRESH")

        # Bottom-left info
        cv2.putText(canvas, yolo_text, (10, 495), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(canvas, taper_text, (10, 515), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(canvas, uneven_text, (10, 535), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Bottom-right result
        (text_w, _), _ = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(canvas, result_text, (630 - text_w, 535), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Live Feed", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
