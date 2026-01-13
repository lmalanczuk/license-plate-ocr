import re
import cv2
import numpy as np
import easyocr


class PlateOcr:
    def __init__(self):
        self.reader = easyocr.Reader(['pl'], gpu=False, verbose=False)

    def _cut_blue_strip(self, img: np.ndarray) -> np.ndarray:
        if img.size == 0:
            return img

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]

        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        scan_limit = int(w * 0.30)
        max_safe_crop = int(w * 0.18)

        cut_location = 0
        in_blue = False

        for x in range(scan_limit):
            density = np.count_nonzero(mask[:, x]) / h
            if density > 0.35:
                in_blue = True
                cut_location = x
            elif in_blue:
                break

        cut_location = min(cut_location, max_safe_crop)

        if cut_location > 0:
            return img[:, cut_location + 2:]

        return img

    def read(self, plate_bgr: np.ndarray) -> str:
        if plate_bgr.size == 0:
            return ""

        h, w = plate_bgr.shape[:2]
        margin = 0.02
        plate = plate_bgr[
            int(h * margin):int(h * (1 - margin)),
            int(w * margin):int(w * (1 - margin))
        ]

        plate = self._cut_blue_strip(plate)

        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        if gray.shape[0] < 60:
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        inv = cv2.bitwise_not(binary)
        contours, _ = cv2.findContours(
            inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        ih, iw = binary.shape[:2]

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if (ch > ih * 0.85 and cw < iw * 0.08) or (cw > iw * 0.4) or (ch < ih * 0.3):
                cv2.drawContours(binary, [cnt], -1, 255, -1)

        # --- CRITICAL FIX ---
        kernel = np.ones((2, 1), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)

        padded = cv2.copyMakeBorder(
            binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255
        )

        results = self.reader.readtext(
            padded,
            detail=0,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )

        return "".join(results)
