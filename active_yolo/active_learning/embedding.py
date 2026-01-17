from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO


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

    results = embedding_model.embed(image_paths, verbose=True, stream=True)

    for image_path, res in zip(image_paths, results):
        embedding = res[0].cpu().numpy()
        embeddings.append(
            ImageEmbeddingResult(
                image_path=image_path,
                embedding=embedding,
            )
        )

    return embeddings
