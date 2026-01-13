import os
import sqlite3
import time
import cv2
import numpy as np

from app.detector import PlateDetector
from app.analyzer import PlateAnalyzer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
DB_PATH = os.path.join(STORAGE_DIR, "results.db")

os.makedirs(STORAGE_DIR, exist_ok=True)

MODEL_PATH = "runs/plate_yolo/weights/best.pt"

detector = PlateDetector(MODEL_PATH)
analyzer = PlateAnalyzer(detector)


def process_image_task(image_bytes: bytes):
    img_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    result = analyzer.analyze(image)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT,
            ok INTEGER,
            ts REAL
        )
    """)

    cur.execute(
        "INSERT INTO results (plate, ok, ts) VALUES (?, ?, ?)",
        (result.get("plate_text"), int(result.get("ok")), time.time())
    )

    conn.commit()
    conn.close()

    return result
