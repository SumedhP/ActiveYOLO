import os
from ultralytics import YOLO, settings  # type: ignore[reportPrivateImportUsage]
from lightly_train import pretrain
from config import AppConfig, TrainConfig
from dataclasses import asdict
import argparse

# Enable TensorBoard logging
settings.update({"tensorboard": True})


def train_model():
    app_config = AppConfig.load_app_config()
    yolo_config = TrainConfig.load_train_config().yolo
    augmentation = yolo_config.augmentation

    # Load the YOLO model
    model = YOLO(yolo_config.model)

    dataset_yaml_path = os.path.join(app_config.dataset_path, "dataset.yaml")

    model.train(
        data=dataset_yaml_path,
        resume=yolo_config.resume,
        epochs=yolo_config.epochs,
        patience=yolo_config.patience,
        batch=yolo_config.batch,
        amp=yolo_config.amp,
        compile=yolo_config.compile,
        device=yolo_config.device,
        optimizer=yolo_config.optimizer,
        lr=yolo_config.lr,
        project="runs",
        **asdict(augmentation),
    )

    model.val()


def train_backbone():
    app_config = AppConfig.load_app_config()
    ssl_config = TrainConfig.load_train_config().ssl

    pretrain(
        out=ssl_config.out_dir,
        model=ssl_config.model,
        data=app_config.raw_images_path,
        epochs=ssl_config.epochs,
        batch_size=ssl_config.batch,
        accelerator=ssl_config.accelerator,
        devices=ssl_config.devices,
        overwrite=ssl_config.overwrite,
        method_args={
            "teacher": ssl_config.teacher,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ssl",
        help="Pretrain the backbone using self-supervised learning",
        action="store_true",
    )
    args = parser.parse_args()

    if args.ssl:
        train_backbone()
    else:
        train_model()
