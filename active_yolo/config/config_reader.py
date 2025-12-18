from dataclasses import dataclass
from typing import Dict, List, Union
import yaml

@dataclass
class TrainConfig:
    # Model path, resume should be true if continuing training
    model: str
    resume: bool
    amp: bool # Whether to use Automatic Mixed Precision - causes weird issues when resuming from a checkpoint
    compile: bool # Whether we should attempt to use PyTorch's model compilation feature

    # Training parameters
    epochs: int
    patience: int
    batch: int
    device: Union[str, List[int]] # 'cpu', [0, 1]


    # Data Augmentation parameters
    hsv_h: float
    hsv_s: float
    hsv_v: float
    degrees: float
    flipud: float
    fliplr: float
    mosaic: float
    scale: float

    @staticmethod
    def load_from_yaml(file_path: str) -> 'TrainConfig':
        with open(file_path, 'r') as f:
            return TrainConfig(**yaml.safe_load(f))

@dataclass
class DataConfig:
    names: Dict[int, str]

    @staticmethod
    def load_from_yaml(path: str) -> "DataConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        # Ensure keys are ints
        data["names"] = {int(k): v for k, v in data["names"].items()}
        return DataConfig(**data)
