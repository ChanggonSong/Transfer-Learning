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

🔹 2.1.5 Batch-wise Prediction vs Ground Truth (Visual Analysis)
To complement numerical evaluation, we visualized prediction results for randomly sampled training and validation batches:

Figure 2.1.5a: train_batch0.jpg to train_batch21062.jpg show ground truth annotations in the training set.

Figure 2.1.5b: val_batch0_labels.jpg, val_batch1_labels.jpg, val_batch2_labels.jpg show validation ground truth labels.

Figure 2.1.5c: val_batch0_pred.jpg, val_batch1_pred.jpg, val_batch2_pred.jpg display model predictions for the same validation samples.

Key Observations:

The model shows consistent performance across batches, especially for frequent classes like shortpants, skirt, and cottonpants.

Confidence scores are generally high, often exceeding 0.85 for clean, well-lit samples (e.g., shortsleeve: 0.93, blazer: 0.97).

Lower performance is observed on rare or visually similar categories (e.g., zipup, padding), aligning with class imbalance insights.

🔹 2.1.6 False Positive / False Negative Examples (Qualitative Errors)
We additionally analyzed ff2dbe3f-bf0a-4bbc-b704-21ad9ca3ea46.jpg to identify qualitative errors:

False Positives: Some predictions identified garments that were not labeled in the ground truth, often due to ambiguous occlusions or complex garment overlaps (e.g., skirt misclassified as shortpants).

False Negatives: In a few cases, true garments were missed, particularly when darker clothing blended into the background or partially occluded (e.g., hoodie under jacket).

These examples demonstrate the need for:

Tuning the confidence threshold,

Post-processing steps like non-maximum suppression adjustment, or

Introducing context-aware modules for better fine-grained apparel detection.

🔹 2.1.7 Label-Prediction Match Consistency
To assess localization and semantic agreement, we directly compared ground truth and predicted labels:

val_batch0_labels.jpg to val_batch2_labels.jpg vs. val_batch0_pred.jpg to val_batch2_pred.jpg.

Findings:

High bounding box alignment was observed (visually consistent IOU), confirming strong spatial learning.

However, semantic confusion occurred among similar garments:

Cardigan ↔ Jacket

Slacks ↔ Cottonpants

This highlights the importance of consistent labeling and perhaps the need for hierarchical class grouping during training.



## 🖼 Inference
yolo predict \
  model=runs/detect/clothing_yolov11/weights/best.pt \
  source=clothes_dataset/images/test \
  save=True


## 📚 Reference
Ultralytics YOLOv11 GitHub

Roboflow Dataset Tools
