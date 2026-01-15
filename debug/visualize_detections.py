import os
import cv2
import xml.etree.ElementTree as ET
import random
from pathlib import Path

from app.detector import PlateDetector

# Konfiguracja
MODEL_PATH = 'runs/plate_yolo/weights/best.pt'
VAL_IMAGES_DIR = "preprocessed_data/images/val"
ANNOTATIONS_XML = "dataset/annotations.xml"
OUTPUT_DIR = "visualization_results"
NUM_SAMPLES = 10  # Ile zdjęć zwizualizować


def load_ground_truth(image_name):
    """Wczytuje ground truth box dla danego zdjęcia z XML."""
    tree = ET.parse(ANNOTATIONS_XML)
    root = tree.getroot()

    for img in root.findall("image"):
        if img.get("name") == image_name:
            box = img.find("box")
            if box is not None:
                attr = box.find(".//attribute[@name='plate number']")
                plate_text = attr.text if attr is not None else "UNKNOWN"

                return {
                    "box": [
                        float(box.get("xtl")),
                        float(box.get("ytl")),
                        float(box.get("xbr")),
                        float(box.get("ybr"))
                    ],
                    "text": plate_text
                }
    return None


def calculate_iou(box1, box2):
    """Oblicza IoU między dwoma bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def draw_box(img, box, color, label, thickness=2):
    """Rysuje bounding box z etykietą."""
    x1, y1, x2, y2 = map(int, box)

    # Rysuj prostokąt
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # Przygotuj tło dla tekstu
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, font_thickness
    )

    # Tło dla tekstu
    cv2.rectangle(
        img,
        (x1, y1 - text_height - baseline - 5),
        (x1 + text_width, y1),
        color,
        -1
    )

    # Tekst
    cv2.putText(
        img,
        label,
        (x1, y1 - baseline),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness
    )


def visualize_detection(image_path, detector):
    """Wizualizuje detekcję dla pojedynczego zdjęcia."""
    # Wczytaj obraz
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️  Nie można wczytać: {image_path}")
        return None

    img_display = img.copy()
    image_name = os.path.basename(image_path)

    # Ground Truth
    gt_data = load_ground_truth(image_name)
    if gt_data is None:
        print(f"⚠️  Brak GT dla: {image_name}")
        return None

    gt_box = gt_data["box"]
    gt_text = gt_data["text"]

    # Predykcja
    pred_box = detector.detect(img)

    # Rysuj Ground Truth (zielony)
    draw_box(img_display, gt_box, (0, 255, 0), f"GT: {gt_text}", thickness=2)

    # Rysuj predykcję (czerwony/niebieski)
    if pred_box is not None:
        iou = calculate_iou(pred_box, gt_box)
        color = (0, 0, 255) if iou < 0.5 else (255, 0, 0)  # Czerwony jeśli IoU < 0.5
        draw_box(img_display, pred_box, color, f"PRED (IoU: {iou:.3f})", thickness=2)
    else:
        # Brak detekcji
        cv2.putText(
            img_display,
            "NO DETECTION!",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

    return img_display


def create_comparison_grid(images, titles, max_cols=2):
    """Tworzy siatkę porównawczą z obrazów."""
    if not images:
        return None

    # Resize wszystkich obrazów do tego samego rozmiaru
    target_height = 400
    resized = []
    for img in images:
        h, w = img.shape[:2]
        ratio = target_height / h
        new_w = int(w * ratio)
        resized.append(cv2.resize(img, (new_w, target_height)))

    # Podziel na rzędy
    rows = []
    for i in range(0, len(resized), max_cols):
        row_imgs = resized[i:i + max_cols]

        # Dodaj padding jeśli niepełny rząd
        while len(row_imgs) < max_cols:
            row_imgs.append(
                255 * np.ones_like(row_imgs[0])
            )

        # Połącz w rząd
        rows.append(cv2.hconcat(row_imgs))

    # Połącz rzędy
    grid = cv2.vconcat(rows)

    return grid


def main():
    print("\n" + "=" * 60)
    print("WIZUALIZACJA DETEKCJI - Model vs Ground Truth")
    print("=" * 60 + "\n")

    # Utwórz folder wyjściowy
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Wczytaj model
    print("📦 Ładowanie modelu...")
    detector = PlateDetector(MODEL_PATH)

    # Pobierz losowe zdjęcia
    val_images = [f for f in os.listdir(VAL_IMAGES_DIR) if f.endswith(('.jpg', '.png'))]

    if len(val_images) == 0:
        print("❌ Brak zdjęć w folderze val!")
        return

    selected = random.sample(val_images, min(NUM_SAMPLES, len(val_images)))

    print(f"🎲 Wylosowano {len(selected)} zdjęć do wizualizacji\n")

    # Przetwarzaj każde zdjęcie
    visualizations = []

    for i, img_name in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] Przetwarzam: {img_name}")

        img_path = os.path.join(VAL_IMAGES_DIR, img_name)
        result = visualize_detection(img_path, detector)

        if result is not None:
            # Zapisz pojedyncze zdjęcie
            output_path = os.path.join(OUTPUT_DIR, f"viz_{img_name}")
            cv2.imwrite(output_path, result)
            visualizations.append(result)
            print(f"   ✅ Zapisano: {output_path}")

    print("\n" + "=" * 60)
    print(f"✅ Wizualizacja zakończona!")
    print(f"📁 Wyniki zapisane w: {OUTPUT_DIR}/")
    print(f"📊 Przetworzono: {len(visualizations)}/{len(selected)} zdjęć")
    print("=" * 60 + "\n")

    # Legenda
    print("LEGENDA:")
    print("🟢 Zielony box    = Ground Truth (prawda)")
    print("🔵 Niebieski box  = Predykcja (IoU >= 0.5)")
    print("🔴 Czerwony box   = Predykcja (IoU < 0.5)")
    print("❌ NO DETECTION   = Model nie znalazł tablicy\n")


if __name__ == "__main__":
    # Wymagane dla cv2.hconcat/vconcat
    import numpy as np

    main()
