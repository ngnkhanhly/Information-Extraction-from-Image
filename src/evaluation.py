import time
import numpy as np
import Levenshtein



# ======================================
# METRICS
# ======================================

def calculate_char_accuracy(pred, gt):
    """Character-level accuracy using Levenshtein distance"""
    pred = pred.lower()
    gt = gt.lower()

    if len(gt) == 0:
        return 1.0 if len(pred) == 0 else 0.0

    distance = Levenshtein.distance(pred, gt)
    acc = 1.0 - distance / max(len(pred), len(gt))

    return max(0.0, acc)


def calculate_word_accuracy(pred, gt):
    """Exact word match accuracy"""
    return float(pred.strip().lower() == gt.strip().lower())


def calculate_metrics(predictions, ground_truths):
    """Aggregate OCR metrics"""
    char_accs = []
    word_accs = []

    for pred, gt in zip(predictions, ground_truths):
        char_accs.append(calculate_char_accuracy(pred, gt))
        word_accs.append(calculate_word_accuracy(pred, gt))

    return {
        "char_accuracy": np.mean(char_accs) * 100 if char_accs else 0.0,
        "word_accuracy": np.mean(word_accs) * 100 if word_accs else 0.0,
        "total_samples": len(predictions),
    }


# ======================================
# IOU MATCHING
# ======================================

def _compute_iou(box1, box2):
    """Compute IoU for (x, y, w, h) format"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    inter_area = (xi2 - xi1) * (yi2 - yi1)
    union_area = w1 * h1 + w2 * h2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def match_predictions_to_ground_truth(
    predictions, ground_truths, iou_threshold=0.3
):
    """Match predicted boxes to GT boxes using IoU"""
    matched_preds = []
    matched_gts = []

    for gt in ground_truths:
        best_iou = 0.0
        best_pred = None

        for pred in predictions:
            iou = _compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_pred = pred

        if best_iou >= iou_threshold and best_pred is not None:
            matched_preds.append(best_pred["text"])
            matched_gts.append(gt["label"])

    return matched_preds, matched_gts


# ======================================
# EVALUATION LOOP
# ======================================

def evaluate_model(
    inference_func,
    test_samples,
    model_name,
    conf_threshold=0.3,
):
    """Evaluate OCR model on test set"""
    print("\n" + "=" * 60)
    print(f"Evaluating {model_name}")
    print("=" * 60)

    all_predictions = []
    all_ground_truths = []

    total_time = 0.0
    processed_images = 0

    # Group GT by image
    images_dict = {}
    for item in test_samples:
        img_path = item["image_path"]
        images_dict.setdefault(img_path, []).append(item)

    for img_path, gt_items in images_dict.items():
        try:
            start = time.time()
            predictions = inference_func(
                img_path, conf_threshold=conf_threshold
            )
            elapsed = time.time() - start

            total_time += elapsed
            processed_images += 1

            matched_preds, matched_gts = match_predictions_to_ground_truth(
                predictions, gt_items
            )

            all_predictions.extend(matched_preds)
            all_ground_truths.extend(matched_gts)

        except Exception as e:
            print(f"[ERROR] {img_path}: {e}")
            continue

    if not all_predictions:
        print(f"No matched predictions for {model_name}")
        return None

    metrics = calculate_metrics(all_predictions, all_ground_truths)
    avg_time = total_time / processed_images if processed_images else 0.0

    print(f"\nResults for {model_name}:")
    print(f"  - Character Accuracy : {metrics['char_accuracy']:.2f}%")
    print(f"  - Word Accuracy      : {metrics['word_accuracy']:.2f}%")
    print(f"  - Avg Time / Image   : {avg_time:.4f}s")
    print(f"  - Images Processed   : {processed_images}")
    print(f"  - Matched Regions    : {len(all_predictions)}")

    return {
        "model": model_name,
        "char_acc": metrics["char_accuracy"],
        "word_acc": metrics["word_accuracy"],
        "avg_time": avg_time,
        "total_images": processed_images,
        "matched_regions": len(all_predictions),
    }
