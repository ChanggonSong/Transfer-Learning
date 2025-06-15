import cv2
from ultralytics import YOLO

# model = YOLO("yolo11n.pt") # original yolov11
model = YOLO(r'weights_only_head\best.pt') # transfer-learning (FC layer)


cap = cv2.VideoCapture(0) # 0: your default camera
print("Opening Camera...")
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
print("Camera opened successfully.")
print("Start inference loop...")

while True: # Real-time detection until key 'q'.
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame)

    # YOLO visualization (bbox, label, conf)
    if results:
        annotated_frame = results[0].plot()
        cv2.imshow('YOLOv11 Real-time Detection', annotated_frame)
    else:
        cv2.imshow('YOLOv11 Real-time Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
