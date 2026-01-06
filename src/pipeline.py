import json
import cv2
import torch
from PIL import Image

from src.recognition import decode


device = "cuda" if torch.cuda.is_available() else "cpu"


# ==================================================
# YOLO + CRNN
# ==================================================

def inference_yolo_crnn(
    img_path,
    yolo_det,
    crnn_transform,
    crnn_inference,
    idx_to_char,
    conf_threshold=0.3,
):
    """Inference using YOLO + CRNN pipeline"""
    predictions = []

    results = yolo_det(img_path, verbose=False, conf=conf_threshold)
    detections = json.loads(results[0].to_json())

    img = cv2.imread(img_path)
    if img is None:
        return predictions

    for det in detections:
        confidence = det["confidence"]
        if confidence < conf_threshold:
            continue

        box = det["box"]
        x1, y1 = int(box["x1"]), int(box["y1"])
        x2, y2 = int(box["x2"]), int(box["y2"])

        cropped = img[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        cropped_pil = Image.fromarray(
            cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        )

        img_tensor = crnn_transform(cropped_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = crnn_inference(img_tensor)
            preds = logits.permute(1, 0, 2).argmax(2)
            text = decode(preds, idx_to_char)[0]

        predictions.append(
            {
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "text": text,
                "confidence": confidence,
            }
        )

    return predictions


# ==================================================
# YOLO + TrOCR
# ==================================================

def inference_yolo_trocr(
    img_path,
    yolo_det,
    trocr_processor,
    trocr_model,
    conf_threshold=0.3,
):
    """Inference using YOLO + TrOCR pipeline"""
    predictions = []

    results = yolo_det(img_path, verbose=False, conf=conf_threshold)
    detections = json.loads(results[0].to_json())

    img = cv2.imread(img_path)
    if img is None:
        return predictions

    for det in detections:
        confidence = det["confidence"]
        if confidence < conf_threshold:
            continue

        box = det["box"]
        x1, y1 = int(box["x1"]), int(box["y1"])
        x2, y2 = int(box["x2"]), int(box["y2"])

        cropped = img[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        cropped_pil = Image.fromarray(
            cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        )

        pixel_values = trocr_processor(
            cropped_pil, return_tensors="pt"
        ).pixel_values.to(device)

        with torch.no_grad():
            generated_ids = trocr_model.generate(pixel_values)
            text = trocr_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

        predictions.append(
            {
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "text": text,
                "confidence": confidence,
            }
        )

    return predictions


# ==================================================
# EasyOCR (end-to-end)
# ==================================================

def inference_easyocr(img_path, easyocr_reader, conf_threshold=0.3):
    """Inference using EasyOCR (detection + recognition)"""
    predictions = []

    results = easyocr_reader.readtext(img_path)

    for bbox, text, confidence in results:
        if confidence < conf_threshold:
            continue

        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]

        x1, y1 = int(min(x_coords)), int(min(y_coords))
        x2, y2 = int(max(x_coords)), int(max(y_coords))

        predictions.append(
            {
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "text": text,
                "confidence": confidence,
            }
        )

    return predictions
