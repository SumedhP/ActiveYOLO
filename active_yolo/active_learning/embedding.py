from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO
from tqdm import tqdm


@dataclass
class ImageEmbeddingResult:
    image_path: str
    embedding: np.ndarray

    def __lt__(self, other):
        return self.image_path < other.image_path

    def __hash__(self):
        return hash(self.image_path)


def compute_embeddings(
    model_path: str, image_paths: List[str]
) -> List[ImageEmbeddingResult]:
    embedding_model = YOLO(model_path)
    embeddings = []

    for image_path in tqdm(image_paths, desc="Computing embeddings", unit="images"):
        result = embedding_model.embed(image_path, verbose=False)
        embedding = result[0].cpu().numpy()
        embeddings.append(ImageEmbeddingResult(image_path, embedding))

    return embeddings
