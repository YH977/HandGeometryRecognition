# Hand Geometry Recognition Using Deep Learning

Biometric person identification from hand-geometry images, built by fine-tuning and
benchmarking five CNN architectures with transfer learning. Best model (InceptionV3)
reaches **98.6% test accuracy** across 40 individuals.

## Key Results

| Model            | Framework            | Test Accuracy |
|------------------|----------------------|---------------|
| **InceptionV3**  | PyTorch              | **98.6%**     |
| EfficientNetV2-S | PyTorch + timm       | 97.3%         |
| VGG16            | TensorFlow / Keras   | 94.1%         |
| MobileNetV2      | PyTorch + timm       | 90.5%         |
| ResNet18         | PyTorch + timm       | 72.7%         |

Best model: **InceptionV3 — 98.6% test accuracy** on a held-out test set.

## Dataset

- **Source:** [dataset name / link]
- **Classes:** 40 individuals
- **Total images:** 1,197 — 818 train / 159 validation / 220 test
- **Input:** RGB hand images, ImageNet-normalized

## Methodology

**Architectures benchmarked:** VGG16, ResNet18, MobileNetV2, InceptionV3, EfficientNetV2-S.
Four were fine-tuned in PyTorch (ResNet18, MobileNetV2, EfficientNetV2-S via the `timm`
library; InceptionV3 via torchvision); VGG16 was fine-tuned in TensorFlow/Keras.

**Data augmentation:** random resized crop, horizontal flip, rotation, color jitter,
and random erasing, with ImageNet mean/std normalization.

**Training techniques:**
- Transfer learning from ImageNet-pretrained weights
- Label smoothing (smoothing = 0.1) on most models
- Cosine-annealing learning-rate schedules (PyTorch) / ReduceLROnPlateau (VGG16)
- Class weighting (balanced) on VGG16 to handle class imbalance
- Early stopping / model checkpointing

**Training configuration:**

| Setting       | Value                                                          |
|---------------|--------------------------------------------------------------- |
| Batch size    | 32                                                             |
| Epochs        | 50 (ResNet18: 70; VGG16: staged 50 + 30)                       |
| Optimizer     | Adam / AdamW, lr = 1e-4 (VGG16: 1e-5 → 1e-4)                   |
| LR scheduler  | CosineAnnealingLR/CosineAnnealingWarmRestarts; ReduceLROnPlateau (VGG16)|
| Loss          | Label-smoothing cross-entropy (InceptionV3: standard CE)       |
| Hardware      | GPU on Google Colab                                            |

**Evaluation:** per-model confusion matrices, accuracy/loss curves, and comparative
benchmarking across all five architectures.

> Note: test-loss values aren't directly comparable across models because most used
> label-smoothing loss while InceptionV3 used standard cross-entropy. Accuracy is the
> primary comparison metric.

## Tech Stack

PyTorch, timm, TensorFlow/Keras, scikit-learn, NumPy, Matplotlib — trained on Google Colab (GPU).

## Repository Structure

.

├── notebooks/

│   ├── VGG16_BestOne.ipynb

│   ├── ResNet18.ipynb

│   ├── MobileNetV2.ipynb

│   ├── InceptionV3.ipynb

│   └── EfficientNetV2.ipynb

├── results/        # confusion matrices, accuracy/loss plots

└── README.md

## Installation

```bash
git clone https://github.com/[your-username]/[repo-name].git
cd [repo-name]
pip install -r requirements.txt
```

## Future Work

- evaluate on a larger or cross-session dataset
- deploy the best model as a REST API for live inference

## Author

Youssef Hany — linkedin.com/in/youssef-hany-46a23734a / github.com/YH977 / yh97767@gmail.com
