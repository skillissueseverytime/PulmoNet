# PulmoNet 🫁

A deep learning model for classifying chest X-ray images into three categories: **Normal**, **Pneumonia**, and **Tuberculosis (TB)** using a custom CNN architecture built with PyTorch.

## Model Architecture

PulmoNet uses a custom CNN (`Medical`) with the following structure:

- **Block 1**: Conv2d (1 → 16) → ReLU → MaxPool2d
- **Block 2**: Conv2d (16 → 32) → ReLU → MaxPool2d
- **Block 3**: Conv2d (32 → 64) → ReLU → MaxPool2d
- **Block 4**: AdaptiveAvgPool2d → Flatten → Linear (64 → 3)

**Input**: 128×128 grayscale chest X-ray images  
**Output**: 3 classes — Normal, Pneumonia, TB

## Training Details

| Parameter         | Value                      |
|-------------------|----------------------------|
| Optimizer         | Adam (lr=0.003)            |
| Loss Function     | CrossEntropyLoss (weighted)|
| Epochs            | 20                         |
| Batch Size        | 32                         |
| Image Size        | 128 × 128 (Grayscale)     |
| Data Augmentation | RandomHorizontalFlip, RandomRotation (±10°) |
| Normalization     | mean=0.5, std=0.5          |

## Results

### 🏆 Highest Accuracy

| Metric                      | Accuracy   |
|-----------------------------|------------|
| **Best Validation Accuracy** | **76.13%** |
| **Test Accuracy**            | **74.45%** |

### 📊 Classification Report

```
              precision    recall  f1-score   support

      Normal       0.64      0.71      0.67       925
   Pneumonia       0.78      0.96      0.86       580
          TB       0.85      0.66      0.74      1064

    accuracy                           0.75      2569
   macro avg       0.76      0.78      0.76      2569
weighted avg       0.76      0.75      0.74      2569
```

### 🔢 Confusion Matrix

|               | Predicted Normal | Predicted Pneumonia | Predicted TB |
|---------------|:----------------:|:-------------------:|:------------:|
| **Actual Normal**    |       654        |         160         |     111      |
| **Actual Pneumonia** |        11        |         556         |      13      |
| **Actual TB**        |       352        |          10         |     702      |

> **Note**: The confusion matrix values are extracted from the trained model's evaluation on the test set (2,569 samples).

## Dataset Structure

```
data1/
├── train/
│   ├── Normal/
│   ├── Pneumonia/
│   └── TB/
├── val/
│   ├── Normal/
│   ├── Pneumonia/
│   └── TB/
└── test/
    ├── Normal/
    ├── Pneumonia/
    └── TB/
```

## Requirements

- Python 3.x
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- seaborn
- numpy
- pandas

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/skillissueseverytime/pulmonet.git
   cd pulmonet
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision scikit-learn matplotlib seaborn numpy pandas
   ```

3. Place your chest X-ray dataset in the `data1/` directory following the structure above.

4. Open and run `chest.ipynb` to train/evaluate the model.

## License

This project is for educational and research purposes.
