import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import yaml


class YoloPreprocessor:
    def __init__(
            self,
            raw_dataset_dir: str = "dataset",
            output_dir: str = "preprocessed_data",
            train_ratio: float = 0.7,
            class_name: str = "plate",
            random_seed: int = 42,
    ):
        self.raw_dataset_dir = Path(raw_dataset_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.class_name = class_name
        self.random_seed = random_seed

        self.images_src = self.raw_dataset_dir / "photos"
        self.xml_path = self.raw_dataset_dir / "annotations.xml"

        self.images_out = self.output_dir / "images"
        self.labels_out = self.output_dir / "labels"

    def prepare(self):
        print("\n" + "=" * 50)
        print("YOLO PREPROCESSING")
        print("=" * 50)

        self._clean_output()
        self._create_dirs()

        samples = self._parse_annotations()
        self._split_and_save(samples)
        self._generate_dataset_yaml()

        print("=" * 50)
        print("Preprocessing finished successfully.")
        print("=" * 50 + "\n")


    def _clean_output(self):
        if self.output_dir.exists():
            print(f"Cleaning existing output directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)

    def _create_dirs(self):
        for p in [
            self.images_out / "train",
            self.images_out / "val",
            self.labels_out / "train",
            self.labels_out / "val",
        ]:
            p.mkdir(parents=True, exist_ok=True)

    def _parse_annotations(self):
        print(f"Parsing annotations from: {self.xml_path}")
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        samples = []

        for img in root.findall("image"):
            name = img.get("name")
            img_path = self.images_src / name

            if not img_path.exists():
                continue

            boxes = [
                box for box in img.findall("box")
                if box.get("label") == self.class_name
            ]

            if not boxes:
                continue

            samples.append((name, boxes))

        if not samples:
            raise RuntimeError("No valid samples found in annotations.xml")

        print(f"Found {len(samples)} valid samples")
        return samples

    def _split_and_save(self, samples):
        random.seed(self.random_seed)
        random.shuffle(samples)

        split_idx = int(len(samples) * self.train_ratio)

        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]

        print(
            f"Split: {len(train_samples)} train, {len(val_samples)} val ({len(val_samples) / len(samples) * 100:.1f}% val)")

        for idx, (name, boxes) in enumerate(samples):
            subset = "train" if idx < split_idx else "val"

            src_img = self.images_src / name
            dst_img = self.images_out / subset / name
            shutil.copy(src_img, dst_img)

            image = cv2.imread(str(src_img))
            h, w = image.shape[:2]

            label_path = self.labels_out / subset / f"{Path(name).stem}.txt"

            with open(label_path, "w") as f:
                for box in boxes:
                    xtl = float(box.get("xtl"))
                    ytl = float(box.get("ytl"))
                    xbr = float(box.get("xbr"))
                    ybr = float(box.get("ybr"))

                    cx = ((xtl + xbr) / 2) / w
                    cy = ((ytl + ybr) / 2) / h
                    bw = (xbr - xtl) / w
                    bh = (ybr - ytl) / h

                    if not all(0.0 <= v <= 1.0 for v in (cx, cy, bw, bh)):
                        continue

                    f.write(f"0 {cx} {cy} {bw} {bh}\n")

        print(f"Images and labels copied to {self.output_dir}")

    def _generate_dataset_yaml(self):
        data = {
            "path": str(self.output_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": 1,
            "names": [self.class_name],
        }

        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

        print(f"Dataset YAML created: {yaml_path}")