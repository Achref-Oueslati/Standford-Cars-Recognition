# Standford-Cars-Recognition
# **Car Image Classification Using Deep Learning**

This project implements a deep learning solution for classifying car images into one of 20 classes. It uses both a custom Convolutional Neural Network (CNN) and a Transfer Learning approach with VGG16 to achieve accurate predictions on the dataset.

---

## **Dataset**
- **Source**: The dataset contains **1,652 images** of 196 class but for this project i only used and organized 20 classes, each representing a car model (e.g., *GMC Canyon Extended Cab 2012*, *Lamborghini Gallardo LP 570-4 Superleggera 2012*).
- **Structure**: Each class is stored in a separate folder, with images in `.jpg` format.

---

## **Project Workflow**
### **1. Dataset Preparation**
- The dataset is loaded directly from its directory using TensorFlow's `image_dataset_from_directory`.
- The data is split into:
  - **70% Training**
  - **20% Validation**
  - **10% Testing**
- Images are resized to `224x224` and normalized to speed up training.

### **2. Models**
#### **Custom CNN**
- A Convolutional Neural Network (CNN) built from scratch:
  - 3 convolutional layers with ReLU activation and max pooling.
  - Dropout layers for regularization.
  - A dense layer with 512 neurons and a softmax output layer.
- Trained for 50 epochs with the Adam optimizer and sparse categorical cross-entropy loss.

#### **Transfer Learning with VGG16**
- A pre-trained VGG16 model (from ImageNet) fine-tuned for this classification task:
  - The top layers are replaced with a flatten layer, a dropout layer, and a dense softmax output layer.
  - Fine-tuning is applied to the last 4 layers of VGG16.
- Data augmentation (random flips, rotations, and zooms) is applied to improve generalization.

### **3. Evaluation**
- Both models are evaluated on the test set using accuracy and a classification report.

---

## **Results**
| **Model**              | **Training Accuracy** | **Validation Accuracy** | **Test Accuracy** |
|--------------------------|-----------------------|--------------------------|--------------------|
| **Custom CNN**           | 99.81%               | 65.62%                  | 65%               |
| **VGG16 (Transfer Learning)** | 96.76%               | 88.75%                  | ~88%              |

- The Transfer Learning approach significantly outperforms the custom CNN, demonstrating the strength of pre-trained models for small datasets.
---

## **Technologies Used**
- **Python**: Programming language.
- **TensorFlow/Keras**: Deep learning framework.
- **Matplotlib/Seaborn**: Visualization libraries.
- **Scikit-learn**: Classification metrics.

---
## **Project Structure**
```
car-classification/
├── CAR DATASET/                # Dataset directory (organized by class)
├── train.py                    # Python script for training models
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── logs/                       # TensorBoard logs for visualization
```

To install the dependencies on a new system:
```
pip install -r requirements.txt
```

## **Future Improvements**
- Expand the dataset with more images to improve generalization.
- Experiment with deeper custom CNN architectures.
- Explore other pre-trained models like ResNet or MobileNet.
- Use advanced regularization techniques like L2 and early stopping.
---
