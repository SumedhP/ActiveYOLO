from dataclasses import dataclass
import os

import yaml


@dataclass
class ActiveLearningConfig:
    model: str  # Model to use for active learning
    images_per_iteration: int  # Number of images to send to human
    num_clusters: int  # For K-means clustering of embeddings
    output_file_name: str
    embeddings_file_name: str

@dataclass
class InferenceConfig:
    confidence_threshold: float
    agnostic_nms: bool
    half: bool  # Use FP16 half precision
    image_size: int # Higher values can result in better accuracy but slower inference


@dataclass
class AppConfig:
    images_path: str
    labels_path: str
    dataset_path: str
    output_path: str

    active_learning: ActiveLearningConfig
    inference: InferenceConfig

    @property
    def imageset_images_path(self) -> str:
        """Path to unlabeled imageset for active learning."""
        return os.path.join(self.images_path, "imageset")
    
    @property
    def validation_images_path(self) -> str:
        """Path to validation set images."""
        return os.path.join(self.images_path, "validation")
    
    @property
    def imageset_labels_path(self) -> str:
        """Path to imageset labels."""
        return os.path.join(self.labels_path, "imageset")
    
    @property
    def validation_labels_path(self) -> str:
        """Path to validation set labels."""
        return os.path.join(self.labels_path, "validation")

    @staticmethod
    def load_from_yaml(file_path: str) -> "AppConfig":
        with open(file_path, "r") as f:
            cfg = yaml.safe_load(f)
        cfg["active_learning"] = ActiveLearningConfig(**cfg["active_learning"])
        cfg["inference"] = InferenceConfig(**cfg["inference"])
        return AppConfig(**cfg)

    @staticmethod
    def load_app_config() -> "AppConfig":
        return AppConfig.load_from_yaml("configs/app.yaml")
