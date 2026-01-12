import re
import cv2
import numpy as np
import easyocr


class PlateOcr:
    def __init__(self):
        self.reader = easyocr.Reader(['pl'], gpu=False, verbose=False)

    def read(self, plate_bgr: np.ndarray) -> str:
        if plate_bgr.size == 0:
            return ""

        gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)

        if gray.shape[0] < 60:
            gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        inv = cv2.bitwise_not(binary)
        contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = binary.shape[:2]

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if (ch > h * 0.85 and cw < w * 0.08) or (cw > w * 0.4) or (ch < h * 0.3):
                cv2.drawContours(binary, [cnt], -1, 255, -1)

        kernel = np.ones((2, 1), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)

        padded = cv2.copyMakeBorder(
            binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255
        )

        results = self.reader.readtext(
            padded,
            detail=0,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        raw = "".join(results)
        return re.sub(r"[^A-Z0-9]", "", raw.upper())
