# 🩺 Pneumonia vs. Normal Classifier with MLflow

A PyTorch-based deep learning project for classifying chest X-ray images into **Pneumonia** or **Normal**, integrated with **MLflow** for experiment tracking and checkpointing.

---

## 🧠 Overview

This project trains and evaluates a **DenseNet121 (CheXNet variant)** model to detect pneumonia from grayscale chest X-ray images. It logs parameters, metrics, and model checkpoints using MLflow.

---

## 📁 Project Structure

```
pneumonia_vs_normal_with_mlflow/
├── Notebooks/
│   ├── Preprocessing/
│       ├── Data_Resizing.ipynb
│       ├── label_encoder.ipynb
│   ├── train/
│       ├── alex_net.ipynb
│       ├── chex_net.ipynb
│       ├── dense_net.ipynb
│       ├── Loading_data.ipynb   
├── models/
├── train_preprocessed.csv
├── README.md
├── .gitignore
└── requirements.txt
```

---

## 🧪 Key Features

- ✅ Binary image classification (Normal vs. Pneumonia)
- 🧠 DenseNet121, DenseNet201 and Alexnet architectures with grayscale input
- 🧼 Data preprocessing and stratified splitting
- 📈 MLflow logging for metrics, loss curves, and F1 scores
- 🧠 Early stopping to prevent overfitting
- 💾 Checkpoint saving and loading

---

## 🛠 Installation

### 1. Clone the Repository

```
git clone https://github.com/MarwanAbdellah/pneumonia_vs_normal_with_mlflow.git
cd pneumonia_vs_normal_with_mlflow
```

### 2. Create and Activate a Virtual Environment

```
python -m venv venv
```

- **Windows**: `venv\Scripts\activate`  
- **macOS/Linux**: `source venv/bin/activate`

### 3. Install Requirements

```
pip install -r requirements.txt
```

---

## 🚀 Training the Model

```
python train.py
```

You can also use the provided Jupyter Notebooks for an interactive experience.

---

## 📊 MLflow Tracking

To start the MLflow UI:

```
mlflow ui
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 🧪 Data Format

Make sure your CSVs include the following format:

```
img_path,clas,split
images/normal1.jpg,NORMAL,train
images/pneumonia1.jpg,PNEUMONIA,train
...
```

---

## 📦 Requirements

```
torch
torchvision
mlflow
pandas
scikit-learn
matplotlib
tqdm
opencv-python
```

Install via:

```
pip install -r requirements.txt
```

---

## 📌 .gitignore Suggestions

```
__pycache__/
.venv/
.env
*.pt
*.safetensors
results/checkpoint-*/
mlruns/
```

---

## 📬 Contact

**Marwan Abdellah**  
📧 [marawan.abdellah0@gmail.com](mailto:marawan.abdellah0@gmail.com)  
🔗 GitHub: [@MarwanAbdellah](https://github.com/MarwanAbdellah)
