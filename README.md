---

# 📄 Information-Extraction-from-Image

**Two-stage OCR Pipeline: YOLO + CRNN**

---

## 📌 Introduction

This project implements a **two-stage Optical Character Recognition (OCR) pipeline** for **text extraction from images**, consisting of:

1. **Text Detection** using YOLO
2. **Text Recognition** using CRNN + CTC
3. **Benchmarking and comparison** with other popular OCR pipelines

The project is designed for **research and experimental analysis**, focusing on:

* Accuracy comparison between OCR architectures
* Trade-offs between **recognition accuracy and inference speed**
* OCR performance analysis on **NVIDIA T4 GPU**

---

## 📂 Project Structure

```bash
Information-Extraction-from-Image/
│
├── datasets/
│   └── SceneTrialTrain/
│       ├── words.xml
│       └── *.jpg
│
├── src/
│   ├── detection.py        # XML → YOLO format
│   ├── recognition.py     # CRNN + CTC
│   ├── pipeline.py        # Inference pipelines
│   └── evaluation.py      # Metrics & evaluation
│
├── model/
│   ├── yolo/
│   │   └── best.pt
│   └── cnn/
│       └── ocr_crnn.pt
│
├── cache/
│   └── val_data.pkl
│
├── yolo_data/
│   ├── train/
│   └── val/
│
├── notebook/
│   ├── 01_text_detection_training.ipynb
│   ├── 02_text_recognition_training.ipynb
│   └── 03_model_comparision.ipynb
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Environment Setup

```bash
pip install -r requirements.txt
```

> 🔧 **Recommended**: Run on **Google Colab with NVIDIA T4 GPU**
> Local execution is mainly for debugging or lightweight inference.

---

## ▶️ How to Run the Project (Colab vs Local)

The project supports **two execution modes**, controlled by the following flag:

```python
USE_COLAB = True
```

This flag determines how **Google Drive is mounted** and how `PROJECT_ROOT` is defined.

---

## 🟢 Running on Google Colab (USE_COLAB = True) — Recommended

### 1️⃣ Prepare the Project Folder

* Download the **entire project folder**
* Upload the **whole folder** to Google Drive, for example:

```text
MyDrive/
└── Information-Extraction-from-Image/
    ├── datasets/
    ├── src/
    ├── model/
    ├── notebook/
    └── requirements.txt
```

⚠️ **Important**
Do **not** upload individual files.
Always upload the **entire project directory** to preserve the folder structure.

---

### 2️⃣ Notebook Configuration

Keep the following code **unchanged** in all notebooks:

```python
USE_COLAB = True

if USE_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    PROJECT_ROOT = "/content/drive/MyDrive/Information-Extraction-from-Image"
else:
    PROJECT_ROOT = os.path.abspath(".")
```

Expected output:

```text
PROJECT_ROOT: /content/drive/MyDrive/Information-Extraction-from-Image
```

---

### 3️⃣ Run the Notebooks (in order)

```text
notebook/01_text_detection_training.ipynb
notebook/02_text_recognition_training.ipynb
notebook/03_model_comparision.ipynb
```

Enable GPU:

```text
Runtime → Change runtime type → GPU (NVIDIA T4)
```

---

## 🔵 Running Locally (USE_COLAB = False)

### 1️⃣ Change Configuration

In the notebooks:

```python
USE_COLAB = False
```

Then:

```python
PROJECT_ROOT = os.path.abspath(".")
```

---

### 2️⃣ Local Execution Notes

* Notebooks must be executed from the **project root directory**
* If no GPU is available:

  * CRNN and TrOCR will be **very slow**
  * Training is **not recommended**

---

## 🧠 OCR Pipeline Overview

```text
Input Image
     ↓
YOLO (Text Detection)
     ↓
Crop text regions
     ↓
CRNN + CTC (Text Recognition)
     ↓
OCR Output
```

---

## 🧪 Notebook 01 — Text Detection (YOLO)

📘 `notebook/01_text_detection_training.ipynb`

* Parse annotations from `words.xml`
* Convert bounding boxes to YOLO format
* Train YOLO text detector
* Save the best-performing model

**Output**:

```text
model/yolo/best.pt
```

---

## 🔤 Notebook 02 — Text Recognition (CRNN + CTC)

📘 `notebook/02_text_recognition_training.ipynb`

### Architecture:

* Backbone: **ResNet34**
* Sequence model: **Bi-GRU**
* Loss function: **CTC Loss**

**Output**:

```text
model/cnn/ocr_crnn.pt
```

---

## ⚖️ Notebook 03 — OCR Pipeline Comparison

📘 `notebook/03_model_comparision.ipynb`

### Evaluated Pipelines

| Pipeline               | Detection  | Recognition |
| ---------------------- | ---------- | ----------- |
| **YOLO + CRNN (Ours)** | YOLO       | CRNN        |
| YOLO + TrOCR           | YOLO       | TrOCR       |
| EasyOCR                | End-to-End | EasyOCR     |

### Evaluation Settings

* Confidence threshold: **0.3**
* IoU threshold: **0.3**
* GPU: **NVIDIA T4**

---

## 📊 Experimental Results (NVIDIA T4)

```text
====================================================================================================
COMPARISON RESULTS - CONFIDENCE THRESHOLD = 0.3
====================================================================================================
               Model  Char Acc (%)  Word Acc (%)  Speed (s/img)  Matched Regions
  YOLO + CRNN (Ours)     90.762663     76.923077       1.725653              195
        YOLO + TrOCR     91.533326     76.410256       0.794261              195
EasyOCR (End-to-End)     81.221196     54.464286       0.173435              112

====================================================================================================
SUMMARY
====================================================================================================
Confidence 0.3 - Best Char Acc: YOLO + TrOCR | Best Word Acc: YOLO + CRNN (Ours)
```

---

## 🧠 Result Analysis

* **YOLO + CRNN**

  * Achieves the **highest Word Accuracy**
  * Slowest inference on T4 due to:

    * Bi-GRU (sequential RNN operations)
    * Small-batch inference
    * Frequent CPU–GPU synchronization

* **YOLO + TrOCR**

  * Best **Character Accuracy**
  * Faster than CRNN due to Transformer-based recognition

* **EasyOCR**

  * Fastest inference
  * Significantly lower accuracy

---

## 📌 Conclusion

* **YOLO + CRNN** is suitable when **accuracy is the priority**
* **YOLO + TrOCR** provides the **best balance** between speed and accuracy on T4
* **EasyOCR** is appropriate for **real-time applications** with lower accuracy requirements

---

## 🚀 Future Work

* Replace CRNN with lightweight Transformer OCR
* Batch recognition inference
* ONNX / TensorRT optimization
* Experiments on NVIDIA L4 / A100 GPUs

---

## 👤 Author

* **Name**: *Ly Nguyen*
* **Purpose**: Research / OCR Benchmarking
* **GPU Used**: NVIDIA T4

---
