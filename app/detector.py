import numpy as np
from ultralytics import YOLO


class PlateDetector:
    def __init__(self, model_path: str, conf: float = 0.25):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, image):
        result = self.model.predict(
            image, conf=self.conf, iou=0.5, verbose=False
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return None

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        idx = int(np.argmax(confs))

        return boxes[idx]
