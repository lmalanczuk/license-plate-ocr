import os
from ultralytics import YOLO


class YoloTrainer:

    def __init__(
        self,
        dataset_yaml: str,
        base_model: str = "yolov8n.pt",
        output_dir: str = "runs",
        model_name: str = "plate_yolo",
        device: str = "cpu",
    ):
        self.dataset_yaml = dataset_yaml
        self.base_model = base_model
        self.output_dir = output_dir
        self.model_name = model_name
        self.device = device

    def train(
        self,
        epochs: int = 30,
        img_size: int = 640,
        batch: int = 16,
        workers: int = 4,
    ):
        model = YOLO(self.base_model)

        model.train(
            data=self.dataset_yaml,
            imgsz=img_size,
            epochs=epochs,
            batch=batch,
            workers=workers,
            device=self.device,
            project=self.output_dir,
            name=self.model_name,
        )

        best_model = os.path.join(
            self.output_dir,
            self.model_name,
            "weights",
            "best.pt",
        )

        if not os.path.exists(best_model):
            raise FileNotFoundError("YOLO training finished but best.pt not found.")

        print(f"Training finished. Best model saved at: {best_model}")
        return best_model
