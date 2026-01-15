import os
import cv2
import random
from pathlib import Path

TRAIN_IMAGES = "preprocessed_data/images/train"
TRAIN_LABELS = "preprocessed_data/labels/train"
OUTPUT_DIR = "training_data_visualization"
NUM_SAMPLES = 10


def parse_yolo_label(label_path, img_width, img_height):
    """
    Parsuje plik YOLO .txt i konwertuje do współrzędnych pikseli.
    Format YOLO: class_id center_x center_y width height (wszystko 0-1)
    """
    boxes = []

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id, cx, cy, w, h = map(float, parts)

            # Konwersja do pikseli
            cx *= img_width
            cy *= img_height
            w *= img_width
            h *= img_height

            # Konwersja z center do corners
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            boxes.append([x1, y1, x2, y2])

    return boxes


def visualize_training_sample(image_path, label_path):
    """Wizualizuje dane treningowe - pełny obraz + bounding box."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    img_display = img.copy()

    # Parsuj label
    boxes = parse_yolo_label(label_path, w, h)

    # Rysuj każdy box
    for box in boxes:
        x1, y1, x2, y2 = box

        # Rysuj prostokąt (żółty)
        cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Etykieta
        cv2.putText(
            img_display,
            "plate",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    return img_display


def main():
    print("\n" + "=" * 60)
    print("WIZUALIZACJA DANYCH TRENINGOWYCH YOLO")
    print("=" * 60 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Pobierz wszystkie obrazy
    image_files = [f for f in os.listdir(TRAIN_IMAGES) if f.endswith(('.jpg', '.png'))]

    if not image_files:
        print("❌ Brak obrazów treningowych!")
        return

    selected = random.sample(image_files, min(NUM_SAMPLES, len(image_files)))

    print(f"🎲 Wylosowano {len(selected)} przykładów treningowych\n")

    for i, img_name in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] Przetwarzam: {img_name}")

        img_path = os.path.join(TRAIN_IMAGES, img_name)
        label_path = os.path.join(TRAIN_LABELS, Path(img_name).stem + ".txt")

        if not os.path.exists(label_path):
            print(f"   ⚠️  Brak pliku label: {label_path}")
            continue

        result = visualize_training_sample(img_path, label_path)

        if result is not None:
            output_path = os.path.join(OUTPUT_DIR, f"train_{img_name}")
            cv2.imwrite(output_path, result)
            print(f"   ✅ Zapisano: {output_path}")

    print("\n" + "=" * 60)
    print(f"✅ Wizualizacja zakończona!")
    print(f"📁 Wyniki w: {OUTPUT_DIR}/")
    print("=" * 60 + "\n")

    print("To pokazuje JAK YOLO widzi dane podczas treningu:")
    print("🟡 Żółty box = Ground Truth z pliku .txt")
    print("📸 Pełny obraz = Tak jak YOLO go dostaje\n")


if __name__ == "__main__":
    main()