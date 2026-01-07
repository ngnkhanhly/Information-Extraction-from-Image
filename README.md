
---

# 📄 Information-Extraction-from-Image

**Two-stage OCR Benchmark: YOLO-based Detection + Sequence Recognition**

---

## 📌 Project Overview

This project studies a **two-stage Optical Character Recognition (OCR) pipeline** that decouples **text detection** and **text recognition**, with the goal of analyzing **accuracy–latency trade-offs** across different OCR system designs.

Rather than building a production-ready OCR service, the project focuses on:

* Understanding design choices in OCR pipelines
* Comparing sequence-based and transformer-based recognizers
* Measuring inference efficiency under controlled conditions

The work is conducted as a **research and experimental benchmark** on a fixed dataset and hardware setup.

---

## 🧠 OCR Pipelines Evaluated

The following OCR pipelines are implemented and compared:

1. **YOLO + CRNN (Ours)**

   * Detection: YOLO-based text detector
   * Recognition: CRNN with CTC loss

2. **YOLO + TrOCR**

   * Same detector as above
   * Transformer-based OCR recognizer

3. **EasyOCR (End-to-End baseline)**

   * Off-the-shelf OCR pipeline

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
│   ├── detection.py        # XML parsing & YOLO label generation
│   ├── recognition.py     # CRNN + CTC implementation
│   ├── pipeline.py        # End-to-end OCR inference
│   └── evaluation.py      # Accuracy & speed metrics
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

**Recommended environment**:

* Google Colab
* NVIDIA T4 GPU

Local execution is supported for debugging or lightweight inference but is **not recommended for training** without GPU acceleration.

---

## ▶️ Reproducibility & Execution

The project supports execution on **Google Colab** or **local machines**, controlled by a single configuration flag:

```python
USE_COLAB = True
```

* `USE_COLAB = True`: mount Google Drive and set project root accordingly
* `USE_COLAB = False`: run from local project root directory

All experiments were conducted using the same configuration and hardware setup to ensure fair comparison.

---

## 🧪 Experimental Workflow

### Notebook 01 — Text Detection Training

📘 `notebook/01_text_detection_training.ipynb`

* Parse word-level annotations from `words.xml`
* Convert bounding boxes to YOLO format
* Train a YOLO-based text detector
* Select the best model based on validation performance

**Output**:

```text
model/yolo/best.pt
```

---

### Notebook 02 — Text Recognition Training (CRNN)

📘 `notebook/02_text_recognition_training.ipynb`

**Model architecture**:

* Backbone: ResNet34
* Sequence model: Bidirectional GRU
* Loss: CTC Loss

The recognizer is trained on cropped text regions produced by the detector.

**Output**:

```text
model/cnn/ocr_crnn.pt
```

---

### Notebook 03 — OCR Pipeline Comparison

📘 `notebook/03_model_comparision.ipynb`

This notebook evaluates and compares multiple OCR pipelines under identical settings.

---

## 📏 Evaluation Protocol

* Detection filtering:

  * Confidence threshold: **0.3**
  * IoU threshold: **0.3**
* Character Accuracy:

  * Computed using normalized Levenshtein distance at character level
* Word Accuracy:

  * Exact string match
* Speed:

  * Average inference time per image (seconds/image)
* Hardware:

  * NVIDIA T4 GPU

---

## ⚖️ Benchmarking Notes

* YOLO-based pipelines share the **same detection results** to isolate recognition performance
* TrOCR and EasyOCR use **pre-trained weights** without additional fine-tuning
* The benchmark emphasizes **relative comparison**, not absolute state-of-the-art performance

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
```

**Summary**:

* Best Character Accuracy: **YOLO + TrOCR**
* Best Word Accuracy: **YOLO + CRNN**

---

## 🧠 Result Analysis

* **YOLO + CRNN**

  * Achieves the highest word-level accuracy
  * Slowest inference due to:

    * Sequential Bi-GRU computation
    * Small batch size for cropped text inference
    * Frequent CPU–GPU synchronization

* **YOLO + TrOCR**

  * Best character-level accuracy
  * Faster inference due to parallel transformer decoding

* **EasyOCR**

  * Fastest inference
  * Lower accuracy, especially at word level

From an engineering perspective, the CRNN pipeline is primarily bottlenecked by **sequential sequence modeling**, rather than detection cost.

---

## 📌 Conclusion

* Two-stage OCR pipelines allow flexible trade-offs between accuracy and latency
* CRNN-based recognizers remain competitive in accuracy but suffer from inference inefficiency
* Transformer-based OCR provides a better balance under GPU execution
* End-to-end OCR systems prioritize speed over accuracy

---

## 🚀 Future Work

* Replace CRNN with lightweight transformer-based OCR
* Batch text-region recognition to reduce overhead
* ONNX / TensorRT optimization
* Experiments on newer GPUs (L4, A100)

---

## 👤 Author

* **Name**: Ly Nguyen
* **Project Type**: OCR Research & Benchmarking
* **Focus**: Model comparison and system-level trade-off analysis
* **Hardware**: NVIDIA T4 GPU

---
