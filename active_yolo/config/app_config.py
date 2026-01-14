from dataclasses import dataclass

import yaml


@dataclass
class DatasetConfig:
    val_split: float
    balance_classes: bool


@dataclass
class ActiveLearningConfig:
    model: str  # Model to use for active learning
    images_per_iteration: int  # Number of images to send to human
    num_clusters: int  # For K-means clustering of embeddings
    output_file_name: str


@dataclass
class InferenceConfig:
    confidence_threshold: float
    agnostic_nms: bool
    half: bool  # Use FP16 half precision


@dataclass
class AppConfig:
    raw_images_path: str
    labels_path: str
    dataset_path: str
    output_path: str

    dataset: DatasetConfig
    active_learning: ActiveLearningConfig
    inference: InferenceConfig

    @staticmethod
    def load_from_yaml(file_path: str) -> "AppConfig":
        with open(file_path, "r") as f:
            cfg = yaml.safe_load(f)
        cfg["dataset"] = DatasetConfig(**cfg["dataset"])
        cfg["active_learning"] = ActiveLearningConfig(**cfg["active_learning"])
        cfg["inference"] = InferenceConfig(**cfg["inference"])
        return AppConfig(**cfg)

    @staticmethod
    def load_app_config() -> "AppConfig":
        return AppConfig.load_from_yaml("configs/app.yaml")
