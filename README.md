# 🧥 YOLOv11 Transfer Learning for Clothes Detection

This project applies **transfer learning** on the YOLOv11 (Ultralytics YOLOv8-compatible) object detection model to classify and localize 18 categories of clothing. The model was fine-tuned on a custom-labeled apparel dataset, achieving strong performance on unseen test images.

---

## 📌 Project Overview

- **Base Model**: YOLOv11 (`yolov11n.pt`)
- **Task**: Object detection of fine-grained clothing types
- **Dataset**: 9,517 images (train: 7,482 / val: 1,016 / test: 1,019)
- **Classes**: 18 clothing categories (e.g., hoodie, blazer, cardigan, etc.)
- **Format**: YOLOv5-compatible dataset structure
- **Tools**: Ultralytics CLI, PyTorch, Roboflow

---

## 🗂 Dataset Structure

```
clothes_dataset/
├── images/
│ ├── train/
│ ├── val/
│ └── test/
├── labels/
│ ├── train/
│ ├── val/
│ └── test/
└── data.yaml
```

- Each `.txt` label file follows the format: class_id x_center y_center width height # all values normalized (0~1)


---

## 🧪 Classes

```yaml
names: ['blazer', 'cardigan', 'coat', 'cottonpants', 'denimpants', 'hoodies', 'jacket', 'longsleeve', 'mtm', 'padding', 'shirt', 'shortpants', 'shortsleeve', 'skirt', 'slacks', 'sweater', 'trainingpants', 'zipup']

```

---

## 🚀 Training Command
yolo train \
  model=yolov11n.pt \
  data=clothes_dataset/data.yaml \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  name=clothing_yolov11


## 📈 Evaluation
yolo val \
  model=runs/detect/clothing_yolov11/weights/best.pt \
  data=clothes_dataset/data.yaml \
  split=test


## 🖼 Inference
yolo predict \
  model=runs/detect/clothing_yolov11/weights/best.pt \
  source=clothes_dataset/images/test \
  save=True


## 📚 Reference
Ultralytics YOLOv11 GitHub

Roboflow Dataset Tools
