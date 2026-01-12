import os
import time
import random
import re
import cv2
import xml.etree.ElementTree as ET

from app.detector import PlateDetector
from app.analyzer import PlateAnalyzer


ANNOTATIONS = "dataset/annotations.xml"
IMAGES_DIR = "dataset/photos"
MODEL_PATH = "models/plate_yolo.pt"
TEST_RATIO = 0.30


def normalize(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def plate_match(pred, gt):
    if not pred or not gt:
        return False

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


def calculate_final_grade(acc, t):
    if acc < 60 or t > 60:
        return 2.0
    score = 0.7 * ((acc - 60) / 40) + 0.3 * ((60 - t) / 50)
    return round((2.0 + 3.0 * score) * 2) / 2


def load_data():
    tree = ET.parse(ANNOTATIONS)
    root = tree.getroot()
    data = []

    for img in root.findall("image"):
        box = img.find("box")
        if box is None:
            continue

        attr = box.find(".//attribute[@name='plate number']")
        if attr is None or not attr.text:
            continue

        img_path = os.path.join(IMAGES_DIR, img.get("name"))
        if not os.path.exists(img_path):
            continue

        data.append({
            "path": img_path,
            "gt": normalize(attr.text),
        })

    return data


def main():
    data = load_data()
    if not data:
        print("No valid data loaded from XML.")
        return

    random.shuffle(data)

    test_size = max(1, int(len(data) * TEST_RATIO))
    test = data[:test_size]

    print(f"Test images: {len(test)} / {len(data)}")

    analyzer = PlateAnalyzer(PlateDetector(MODEL_PATH))

    correct = 0
    total = len(test)

    start = time.perf_counter()

    for item in test:
        img = cv2.imread(item["path"])
        if img is None:
            continue

        res = analyzer.analyze(img)

        if not res.get("ok"):
            continue

        if plate_match(res.get("plate_text"), item["gt"]):
            correct += 1

    elapsed = time.perf_counter() - start

    accuracy = correct / total * 100
    time_per_100 = elapsed / total * 100

    print("\n==============================")
    print(f"Correct reads: {correct}")
    print(f"Total test images: {total}")
    print(f"End-to-end accuracy: {accuracy:.2f}%")
    print(f"Time: {time_per_100:.2f}s / 100 images")
    print(f"Grade: {calculate_final_grade(accuracy, time_per_100)}")
    print("==============================\n")


if __name__ == "__main__":
    main()
