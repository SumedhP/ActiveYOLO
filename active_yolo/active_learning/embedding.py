import numpy as np
from ultralytics import YOLO # type: ignore[reportPrivateImportUsage]


def compute_embedding(model: YOLO, image_path: str) -> np.ndarray:
    embedding = model.embed(image_path, verbose=False)[0].numpy()
    return embedding
