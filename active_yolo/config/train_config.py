from dataclasses import dataclass
from typing import List, Union
import yaml


@dataclass
class AugmentationConfig:
    hsv_h: float
    hsv_s: float
    hsv_v: float
    degrees: float
    flipud: float
    fliplr: float
    mosaic: float
    scale: float

@dataclass
class YOLOTrainConfig:
    model: str  # Model path
    resume: bool
    amp: bool  # Can cause issues when resuming from a checkpoint
    compile: bool  # Enable PyTorch model compilation

    # Training parameters
    epochs: int
    patience: int
    batch: int
    device: Union[str, List[int]]  # 'cpu', [0, 1]
    optimizer: str

    augmentation: AugmentationConfig

    @staticmethod
    def load_from_dict(cfg: dict) -> "YOLOTrainConfig":
        cfg["augmentation"] = AugmentationConfig(**cfg["augmentation"])
        return YOLOTrainConfig(**cfg)


@dataclass
class SSLTrainConfig:
    out_dir: str
    model: str

    # Training parameters
    epochs: int
    batch: int

    accelerator: str
    devices: int
    overwrite: bool

    teacher: str

    @staticmethod
    def load_from_dict(cfg: dict) -> "SSLTrainConfig":
        return SSLTrainConfig(**cfg)


@dataclass
class TrainConfig:
    yolo: YOLOTrainConfig
    ssl: SSLTrainConfig

    @staticmethod
    def load_from_yaml(file_path: str) -> "TrainConfig":
        with open(file_path, "r") as f:
            cfg = yaml.safe_load(f)
        cfg["yolo"] = YOLOTrainConfig.load_from_dict(cfg["yolo"])
        cfg["ssl"] = SSLTrainConfig.load_from_dict(cfg["ssl"])
        return TrainConfig(**cfg)

    @staticmethod
    def load_train_config() -> "TrainConfig":
        return TrainConfig.load_from_yaml("configs/train.yaml")
