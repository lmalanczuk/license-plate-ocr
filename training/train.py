from training.yolo_preprocessor import YoloPreprocessor
from training.yolo_trainer import YoloTrainer


def main():
    YoloPreprocessor(
        raw_dataset_dir="dataset",
        output_dir="preprocessed_data",
    ).prepare()

    YoloTrainer(
        dataset_dir="preprocessed_data",
        device="cpu",
    ).train(
        epochs=50,
        batch=16,
        img_size=640,
    )


if __name__ == "__main__":
    main()
