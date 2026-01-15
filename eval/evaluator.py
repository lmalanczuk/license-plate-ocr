import os
import time
import re
import cv2
import xml.etree.ElementTree as ET

from app.detector import PlateDetector
from app.analyzer import PlateAnalyzer

VAL_IMAGES_DIR = "preprocessed_data/images/val"
MODEL_PATH = 'runs/plate_yolo/weights/best.pt'

STRICT_MATCH = True


def normalize(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def plate_match(pred: str, gt: str, strict: bool = True) -> bool:
    pred = normalize(pred)
    gt = normalize(gt)

    if not pred or not gt:
        return False

    if strict:
        return pred == gt

    # Levenshtein distance
    dp = [[0] * (len(gt) + 1) for _ in range(len(pred) + 1)]
    for i in range(len(pred) + 1):
        dp[i][0] = i
    for j in range(len(gt) + 1):
        dp[0][j] = j

    for i in range(1, len(pred) + 1):
        for j in range(1, len(gt) + 1):
            cost = 0 if pred[i - 1] == gt[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    dist = dp[-1][-1]
    similarity = 1 - dist / max(len(pred), len(gt))
    return similarity >= 0.75


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def calculate_final_grade(acc: float, t: float) -> float:
    if acc < 60 or t > 60:
        return 2.0

    score = 0.7 * ((acc - 60) / 40) + 0.3 * ((60 - t) / 50)
    return round((2.0 + 3.0 * score) * 2) / 2


def load_validation_data():
    annotations_xml = "dataset/annotations.xml"

    if not os.path.exists(annotations_xml):
        raise FileNotFoundError(f"Annotations file not found: {annotations_xml}")

    if not os.path.exists(VAL_IMAGES_DIR):
        raise FileNotFoundError(f"Validation images directory not found: {VAL_IMAGES_DIR}")

    tree = ET.parse(annotations_xml)
    root = tree.getroot()

    val_images = set(os.listdir(VAL_IMAGES_DIR))

    data = []

    for img in root.findall("image"):
        name = img.get("name")

        if name not in val_images:
            continue

        box = img.find("box")
        if box is None:
            continue

        attr = box.find(".//attribute[@name='plate number']")
        if attr is None or not attr.text:
            continue

        img_path = os.path.join(VAL_IMAGES_DIR, name)
        if not os.path.exists(img_path):
            continue

        xtl = float(box.get("xtl"))
        ytl = float(box.get("ytl"))
        xbr = float(box.get("xbr"))
        ybr = float(box.get("ybr"))

        data.append({
            "path": img_path,
            "gt": normalize(attr.text),
            "gt_box": [xtl, ytl, xbr, ybr]
        })

    return data


def main():
    data = load_validation_data()

    if not data:
        print("No validation data found!")
        return

    total = len(data)
    print(f"\n{'=' * 50}")
    print(f"EVALUATION ON VALIDATION SET")
    print(f"{'=' * 50}")
    print(f"Validation images: {total}")
    print(f"Match mode: {'STRICT (Exact Match)' if STRICT_MATCH else 'FUZZY (Levenshtein >= 0.75)'}")
    print(f"{'=' * 50}\n")

    print("Loading model...")
    detector = PlateDetector(MODEL_PATH)
    analyzer = PlateAnalyzer(detector)

    correct_ocr = 0
    detected_plates = 0
    total_inference_time = 0.0
    ious = []

    print("Starting evaluation...\n")

    for i, item in enumerate(data, 1):
        img = cv2.imread(item["path"])
        if img is None:
            print(f"⚠️  [{i}/{total}] Could not load image: {item['path']}")
            continue

        t_start = time.perf_counter()

        pred_box = detector.detect(img)

        if pred_box is not None:
            detected_plates += 1
            x1, y1, x2, y2 = map(int, pred_box)
            plate = img[y1:y2, x1:x2]
            raw = analyzer.ocr.read(plate)
            from app.postprocess import smart_postprocess
            final = smart_postprocess(raw)

            # IoU
            iou = calculate_iou(pred_box, item["gt_box"])
            ious.append(iou)

            # OCR match
            if plate_match(final, item["gt"], strict=STRICT_MATCH):
                correct_ocr += 1
                status = "✓"
            else:
                status = "✗"

            if i % 10 == 0 or status == "✗":
                print(f"{status} [{i}/{total}] GT: {item['gt']:10s} | PRED: {final:10s} | IoU: {iou:.3f}")
        else:
            print(f"✗ [{i}/{total}] GT: {item['gt']:10s} | PRED: {'NO_DETECT':10s} | IoU: 0.000")

        t_end = time.perf_counter()
        total_inference_time += (t_end - t_start)

    detection_rate = (detected_plates / total) * 100 if total > 0 else 0
    accuracy = (correct_ocr / total) * 100 if total > 0 else 0
    time_per_100 = (total_inference_time / total) * 100 if total > 0 else 0
    mean_iou = sum(ious) / len(ious) if ious else 0.0

    print(f"\n{'=' * 50}")
    print(f"FINAL RESULTS")
    print(f"{'=' * 50}")
    print(f"Detection Rate:    {detected_plates}/{total} ({detection_rate:.2f}%)")
    print(f"Mean IoU:          {mean_iou:.4f}")
    print(f"Correct OCR:       {correct_ocr}/{total}")
    print(f"End-to-End Acc:    {accuracy:.2f}%")
    print(f"Time (100 imgs):   {time_per_100:.2f}s")
    print(f"Projected Grade:   {calculate_final_grade(accuracy, time_per_100)}")
    print(f"{'=' * 50}\n")

    if accuracy < 60:
        print("WARNING: Accuracy below 60%. Grade will be 2.0.")
    if time_per_100 > 60:
        print("WARNING: Time over 60s/100imgs. Grade will be 2.0.")
    if mean_iou < 0.5:
        print("WARNING: Mean IoU below 0.5. Detection quality may be poor.")


if __name__ == "__main__":
    main()