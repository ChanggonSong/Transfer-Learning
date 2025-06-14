import cv2
from ultralytics import YOLO

# 1. Load model (on your path)
model = YOLO(r'Transfer-Learning\weights_only_head/best.pt')

# 2. webcam initialization
cap = cv2.VideoCapture(0)  # 0: Your Laptop camera
print("Opening Camera...")
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
print("Opening Camera Successed. ")
print("Start inference loop...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # 3. YOLOv11 inference
    results = model(frame)

    # 4. Visualize bbox and label name on real time screen
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"{model.names[cls_id]}: {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # 5. Show result
    cv2.imshow('YOLOv11 Real-time Detection', frame)

    # Quit by 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
