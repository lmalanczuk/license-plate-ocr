import time
import cv2
import numpy as np

from app.detector import PlateDetector
from app.ocr import PlateOcr
from app.postprocess import PlatePostProcessor


class PlateAnalyzer:
    def __init__(self, detector: PlateDetector):
        self.detector = detector
        self.ocr = PlateOcr()
        self.post = PlatePostProcessor()

    def analyze(self, image: np.ndarray) -> dict:
        t0 = time.perf_counter()

        box = self.detector.detect(image)
        if box is None:
            return {"ok": False}

        x1, y1, x2, y2 = map(int, box)
        plate = image[y1:y2, x1:x2]

        raw = self.ocr.read(plate)
        final = self.post.process(raw)

        return {
            "ok": bool(final),
            "plate_text": final,
            "time_ms": int((time.perf_counter() - t0) * 1000)
        }
