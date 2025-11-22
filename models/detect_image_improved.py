# detect_image.py
import cv2
from ultralytics import YOLO
from util import read_license_plate

# load models
vehicle_model = YOLO("models/yolov8n.pt")
plate_model = YOLO("models/license_plate_detector.pt")

# load image
img = cv2.imread("image.png")
if img is None:
    print("image not found")
    exit()

output = img.copy()

# detect vehicles
vehicle_results = vehicle_model(img)[0]

# COCO vehicle classes: car(2), motorcycle(3), bus(5), truck(7)
vehicle_classes = [2, 3, 5, 7]

for det in vehicle_results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = det
    if int(class_id) not in vehicle_classes:
        continue

    # crop vehicle
    vx1, vy1 = int(x1), int(y1)
    vx2, vy2 = int(x2), int(y2)
    vehicle_crop = img[vy1:vy2, vx1:vx2]

    if vehicle_crop.size == 0:
        continue

    # detect plates inside vehicle crop
    plate_results = plate_model(vehicle_crop)[0]

    for p in plate_results.boxes.data.tolist():
        px1, py1, px2, py2, pscore, pclass = p

        # convert relative coords to original image coords
        ax1 = vx1 + int(px1)
        ay1 = vy1 + int(py1)
        ax2 = vx1 + int(px2)
        ay2 = vy1 + int(py2)

        # safety clamp
        ax1 = max(0, ax1)
        ay1 = max(0, ay1)
        ax2 = min(img.shape[1], ax2)
        ay2 = min(img.shape[0], ay2)

        plate_crop = img[ay1:ay2, ax1:ax2]
        if plate_crop.size == 0:
            continue

        # preprocess for OCR
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)

        # read plate text
        text, text_score = read_license_plate(thresh)

        print("plate:", text, "score:", text_score)

        # draw on image
        cv2.rectangle(output, (ax1, ay1), (ax2, ay2), (0, 0, 255), 2)
        label = text if text else "unknown"
        cv2.putText(output, label, (ax1, ay1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

# save output
cv2.imwrite("annotated.jpg", output)
print("saved annotated.jpg")


