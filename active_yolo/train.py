import argparse
import os
from dataclasses import asdict

from config import AppConfig, TrainConfig
from lightly_train import pretrain
from ultralytics import YOLO, settings

# Enable TensorBoard logging
settings.update({"tensorboard": True})


def train_model():
    app_config = AppConfig.load_app_config()
    yolo_config = TrainConfig.load_train_config().yolo

    # Load the YOLO model
    model = YOLO(yolo_config.model)

    dataset_yaml_path = os.path.join(app_config.dataset_path, "dataset.yaml")

    model_dict = asdict(yolo_config)
    model_dict.pop("model", None)
    model_dict["lr0"] = model_dict.pop("lr", None)

    model.train(
        data=dataset_yaml_path,
        **model_dict,
        project="models",
    )

def train_backbone():
    app_config = AppConfig.load_app_config()
    ssl_config = TrainConfig.load_train_config().ssl

    pretrain(
        out=ssl_config.out_dir,
        model=ssl_config.model,
        data=app_config.imageset_images_path,
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
