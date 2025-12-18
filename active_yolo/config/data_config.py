from dataclasses import dataclass
from typing import Dict
import yaml


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

    @staticmethod
    def load_data_config() -> "DataConfig":
        return DataConfig.load_from_yaml("configs/data.yaml")
