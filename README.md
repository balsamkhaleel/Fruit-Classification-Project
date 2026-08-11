# 🍎 Fruit Classification using VGG16

A Computer Vision project for classifying fruit images using image preprocessing techniques and Transfer Learning with the pretrained VGG16 model.

---

## 📌 Project Overview

This project focuses on fruit image classification using a combination of traditional image processing techniques and deep learning.

The workflow includes:

* Image cleaning
* Duplicate detection and removal
* Color enhancement using LAB color space and CLAHE
* Image denoising
* Image resizing
* Pixel normalization
* Train/Test splitting
* Transfer Learning using VGG16
* Model training and evaluation

The final model uses a pretrained **VGG16** network with additional fully connected layers for fruit classification.

---

## 🎯 Objectives

The main objectives of this project are to:

* Clean and prepare the fruit image dataset.
* Detect and remove duplicate images.
* Improve image quality using image processing techniques.
* Enhance local contrast using CLAHE.
* Reduce image noise using denoising techniques.
* Resize images to a consistent size.
* Normalize image pixel values.
* Split the dataset into training and testing sets.
* Apply Transfer Learning using the pretrained VGG16 architecture.
* Evaluate the model on the test dataset.

---

## 🗂️ Dataset

The project uses a fruit image dataset organized into multiple fruit categories.

The dataset is not included in this repository due to its size.

### 📂 Dataset Access

The dataset used for this project is available through the following Google Drive folder:

🔗 **[Fruit Classification Dataset – Google Drive](https://drive.google.com/drive/folders/1RbXpXgNdLF7uWpfIz-5SxCF-AVgWpDoN?usp=drive_link)**

> Please make sure you have the required access permissions to view or download the dataset.

---

## 🧹 Data Cleaning

Before training the model, the images go through a cleaning and preprocessing pipeline.

### Duplicate Removal

Duplicate files are detected based on cleaned filenames.

The filename is processed by removing:

* File extensions
* Non-alphanumeric characters

This helps identify files that may represent duplicates with slightly different naming formats.

---

## 🎨 Image Preprocessing

The following preprocessing pipeline is applied to the images:

```text
Original Image
      ↓
Duplicate Removal
      ↓
LAB Color Space
      ↓
Channel Adjustment
      ↓
CLAHE
      ↓
Denoising
      ↓
Resize
      ↓
Normalization
      ↓
Processed Image
```

### 1. LAB Color Space

Images are converted from BGR to the **LAB color space**.

The LAB representation separates lightness from color information, allowing contrast enhancement to be applied mainly to the lightness component.

---

### 2. CLAHE

**Contrast Limited Adaptive Histogram Equalization (CLAHE)** is applied to the L channel.

Configuration:

```text
clipLimit = 1.5
tileGridSize = (8, 8)
```

The A and B channels are also adjusted before reconstructing the image.

---

### 3. Image Denoising

The project uses **Fast Non-Local Means Denoising** to reduce noise while preserving important image details.

```python
cv2.fastNlMeansDenoisingColored()
```

---

### 4. Image Resizing

During the preprocessing stage, images are resized to:

```text
223 × 223
```

When the processed images are loaded for model training, they are resized to:

```text
224 × 224
```

---

### 5. Normalization

Pixel values are converted to floating-point values and scaled to the range:

```text
0–255 → 0–1
```

The processed images are then converted back to `uint8` before being saved.

---

## ✂️ Train/Test Split

After preprocessing, the images are divided into:

* **80% Training**
* **20% Testing**

The split is performed separately for each fruit category.

```text
Dataset
   │
   ├── 80% → Train
   │
   └── 20% → Test
```

---

## 🧠 Model — VGG16 Transfer Learning

The project uses **VGG16**, a pretrained convolutional neural network with weights obtained from ImageNet.

The pretrained convolutional layers are frozen:

```python
base_model.trainable = False
```

This allows the model to reuse pretrained visual features while training a new classification head for the fruit classification task.

### Model Architecture

```text
Input Image
224 × 224 × 3
      │
      ▼
Pretrained VGG16
      │
      │ Frozen Layers
      ▼
Flatten
      │
      ▼
Dense (256)
      │
      ▼
Dropout (0.5)
      │
      ▼
Softmax
      │
      ▼
Fruit Classes
```

---

## ⚙️ Model Configuration

| Parameter          | Value                     |
| ------------------ | ------------------------- |
| Model              | VGG16                     |
| Pretrained Weights | ImageNet                  |
| Input Size         | 224 × 224 × 3             |
| Base Model         | Frozen                    |
| Dense Layer        | 256 units                 |
| Dropout            | 0.5                       |
| Optimizer          | Adam                      |
| Loss Function      | Categorical Cross-Entropy |
| Batch Size         | 32                        |
| Epochs             | 5                         |
| Output Activation  | Softmax                   |

---

## 📈 Training

The model is trained using the preprocessed training dataset.

During training, the project records:

* Training Accuracy
* Training Loss

Training curves are plotted to visualize the model's learning process.

The trained model is saved as:

```text
vgg16_fruit_transfer.h5
```

> The trained model file is not included in this repository because of its size.

---

## 🧪 Model Evaluation

After training, the saved VGG16 model is loaded and evaluated on the test dataset.

For every test image, the model generates:

* Predicted fruit class
* Prediction confidence

The final test accuracy is calculated using:

```text
Test Accuracy =
Correct Predictions / Total Test Images × 100
```

The project also visualizes the final test accuracy using a bar chart.

---

## 🔄 Complete Workflow

```text
Fruit Dataset
      ↓
Duplicate Detection
      ↓
Duplicate Removal
      ↓
LAB Color Space
      ↓
CLAHE
      ↓
Denoising
      ↓
Resize
      ↓
Normalization
      ↓
80/20 Train-Test Split
      ↓
VGG16 Transfer Learning
      ↓
Model Training
      ↓
Model Saving
      ↓
Test Set Prediction
      ↓
Accuracy Evaluation
```

---

## 🛠️ Technologies & Libraries

* **Python**
* **TensorFlow / Keras**
* **VGG16**
* **OpenCV**
* **NumPy**
* **Matplotlib**
* **Google Colab**

---

## 📁 Project Structure

```text
Fruit-Classification-Project/
│
├── project_cv.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/balsamkhaleel/Fruit-Classification-Project.git
```

### 2. Navigate to the Project Directory

```bash
cd Fruit-Classification-Project
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

The original implementation was developed using **Google Colab** and Google Drive.

### Step 1 — Prepare the Dataset

Download or access the dataset through the provided Google Drive folder:

🔗 **[Fruit Classification Dataset – Google Drive](https://drive.google.com/drive/folders/1RbXpXgNdLF7uWpfIz-5SxCF-AVgWpDoN?usp=drive_link)**

### Step 2 — Configure the Dataset Path

Update the dataset paths in `project_cv.py` according to your Google Drive structure.

### Step 3 — Run the Project

The project can be executed using Google Colab or a compatible Python environment.

```bash
python project_cv.py
```

---

## 📊 Results

The project calculates the final classification accuracy on the test dataset and generates visualizations for:

* Training Accuracy
* Training Loss
* Final Test Accuracy

The exact results depend on the dataset and training run.

---

## 🔮 Future Improvements

Possible improvements for future versions include:

* Fine-tuning the pretrained VGG16 layers.
* Increasing the number of training epochs.
* Adding a dedicated validation dataset.
* Comparing VGG16 with other pretrained architectures such as ResNet, EfficientNet, and MobileNet.
* Applying additional data augmentation techniques.
* Using Precision, Recall, and F1-Score for more detailed evaluation.
* Adding a confusion matrix for class-level performance analysis.
* Performing hyperparameter optimization.
* Deploying the trained model as a web application or API.





