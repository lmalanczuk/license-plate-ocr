import os
from ultralytics import YOLO
from pathlib import Path


class YoloTrainer:
    def __init__(
        self,
        dataset_dir: str = "preprocessed_data",
        base_model: str = "yolov8n.pt",
        device: str = "cpu",
        output_dir: str = "runs",
        model_name: str = "plate_yolo",
    ):
        self.dataset_yaml = Path(dataset_dir) / "dataset.yaml"
        self.base_model = base_model
        self.device = device
        self.output_dir = output_dir
        self.model_name = model_name

        if not self.dataset_yaml.exists():
            raise FileNotFoundError(
                f"dataset.yaml not found at {self.dataset_yaml}"
            )

    def train(
        self,
        epochs: int = 50,
        img_size: int = 640,
        batch: int = 16,
        workers: int = 4,
    ):
        model = YOLO(self.base_model)

        model.train(
            data=str(self.dataset_yaml),
            imgsz=img_size,
            epochs=epochs,
            batch=batch,
            workers=workers,
            device=self.device,
            project=self.output_dir,
            name=self.model_name,
        )

        best_model = (
            Path(self.output_dir)
            / self.model_name
            / "weights"
            / "best.pt"
        )

        if not best_model.exists():
            raise RuntimeError("❌ Training finished but best.pt not found")

        print(f"✅ Training finished. Best model: {best_model}")
        return best_model
