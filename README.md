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

#### 🔹 Evaluation Results
The trained model was evaluated using a combination of quantitative metrics and visual analysis.

This section summarizes the key outcomes from the evaluation phase.


#### 🔹 Confusion Matrix Analysis

To evaluate inter-class confusion, both the absolute and normalized confusion matrices were analyzed:

&nbsp;&nbsp;&nbsp;&nbsp;• Most predictions align well along the diagonal, indicating strong classification accuracy.

&nbsp;&nbsp;&nbsp;&nbsp;• High accuracy was observed for classes such as shortpants, shortsleeve, and denimpants.

&nbsp;&nbsp;&nbsp;&nbsp;• Some confusion was noted among visually similar categories, such as jacket, cardigan, sweater, and longsleeve.

    
Image 1 – Confusion Matrix (absolute counts)
    
Image 2 – Normalized Confusion Matrix (proportions)
    

#### 🔹 Confidence-Based Metric Curves

  Confidence-based evaluation curves were plotted to understand how the model behaves across different confidence thresholds:
    • Most classes show stable precision and recall in the confidence range of 0.7–0.9.  
    • shortsleeve and shortpants achieved exceptionally high scores across all confidence levels.  
    • On the other hand, zipup showed lower performance, likely due to visual overlap with similar items and data scarcity.


    Image 3 – Precision vs. Confidence Curve
    
    Image 4 – F1 Score vs. Confidence Curve
    
    Image 5 – Recall vs. Confidence Curve
    
    Image 6 – Precision-Recall Curve + mAP@0.5 (0.744)
    

#### 🔹 Label Distribution and Bounding Box Analysis

  We also analyzed the label frequency distribution and spatial characteristics of bounding boxes:

    • denimpants, shortpants, and shortsleeve appeared frequently in the dataset, which correlates with their strong detection performance.  
    • Bounding boxes are densely centered in the image frame, and their size distribution shows a healthy variety, reducing risk of spatial bias.


    Image 7 – Class frequency histogram + bbox heatmaps (x/y/width/height)
    
    Image 8 – Bounding box correlation plots (Correlogram)


#### 🔹 Summary of Model Performance

    • The model achieved a mean Average Precision (mAP@0.5) of 0.744 across all classes.  
    • shortpants, shortsleeve, and skirt demonstrated the most robust performance, suggesting practical application potential.  
    • Confusion among jacket-type garments suggests future improvements could involve fine-grained loss functions or more specialized model architectures.


#### 🔹 Batch-wise Prediction vs Ground Truth (Visual Analysis)

  To complement numerical evaluation, we visualized prediction results for randomly sampled training and validation batches:
  
    Figure 2.1.5a: train_batch0.jpg to train_batch21062.jpg show ground truth annotations in the training set.
    
    Figure 2.1.5b: val_batch0_labels.jpg, val_batch1_labels.jpg, val_batch2_labels.jpg show validation ground truth labels.
    
    Figure 2.1.5c: val_batch0_pred.jpg, val_batch1_pred.jpg, val_batch2_pred.jpg display model predictions for the same validation samples.


  Key Observations:
  
    • The model shows consistent performance across batches, especially for frequent classes like shortpants, skirt, and cottonpants.
    
    • Confidence scores are generally high, often exceeding 0.85 for clean, well-lit samples (e.g., shortsleeve: 0.93, blazer: 0.97).
    
    • Rare classes like zipup or padding show lower prediction confidence and are occasionally confused with visually similar categories, reflecting the class imbalance noted in the training distribution.


#### 🔹 False Positive / False Negative Examples (Qualitative Errors)  
  The added figure (ff2dbe3f-bf0a-4bbc-b704-21ad9ca3ea46.jpg) demonstrates common misclassification patterns:
  
    • False Positives: Bounding boxes predicted for garments not present in ground truth, often caused by overlapping or ambiguous clothing views (e.g., shortpants vs. skirt).  
    
    • False Negatives: Some garments annotated in the ground truth were completely missed — especially darker or partially occluded items (e.g., hoodie under jacket).

    This analysis suggests the need for improved post-processing and better threshold tuning in deployment.


#### 🔹 Label-Prediction Match Consistency

  A side-by-side review of val_batch*_labels.jpg and val_batch*_pred.jpg showed:

    • High spatial consistency in predicted bounding boxes (visually aligned with ground truth).
    
    • Moderate confusion between semantically similar categories like cardigan vs. jacket, and slacks vs. cottonpants.
    
    • Emphasizes the need for consistent annotation guidelines when dealing with fine-grained apparel classes.


#### 🔹 Overall Evaluation Summary

The combination of quantitative and qualitative evaluation suggests the YOLO-based clothing detection model is effective across most categories.

  Strengths:

    • Strong generalization in diverse images (pose, lighting, occlusion).
    
    • mAP@0.5 = 0.744 with especially high scores on shortsleeve, shortpants, and skirt.
    
    • High confidence detection for common classes.


  Limitations:

    • Frequent misclassification between similar garments (e.g., jacket vs. cardigan).
    
    • Weak performance on rare or ambiguous classes like zipup, padding.


  Recommendations:

    • Incorporate fine-grained losses or hierarchical class structures.
    
    • Apply focal loss or class reweighting for imbalance.
    
    • Consider vision-language models (e.g., OWL-ViT, YOLO-World) for open-vocabulary detection.


## 📷 Real-time Detection by webcam
To test the trained model in real time using your laptop's webcam:

  1. Install dependencies
    Make sure you have installed all required packages:
    ```
    pip install -r requirements.txt
    ```
  2. Run the script
    Execute the following command in your terminal:
    ```
    python run_yolov11_webcam.py
    ```
  3. Usage
    - A window will open showing your webcam feed with real-time bounding boxes and class labels.
    - Press q to close the window and stop the program.

Note:
  - The webcam index (0) is set for the default laptop camera. Change it to 1 or another index if you use an external camera.
  - Make sure the model weights path in the script matches your actual file location.


## 🖼 Inference
yolo predict \
  model=runs/detect/clothing_yolov11/weights/best.pt \
  source=clothes_dataset/images/test \
  save=True


## 📚 Reference
Ultralytics YOLOv11 GitHub

Roboflow Dataset Tools
