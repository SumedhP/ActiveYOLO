import multiprocessing
from dataclasses import dataclass
from functools import partial
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO  # type: ignore[reportPrivateImportUsage]


@dataclass
class ImageEmbeddingResult:
    image_path: str
    embedding: np.ndarray

    def __hash__(self):
        return hash(self.image_path)


def _compute_embeddings(
    model_path: str, image_paths: List[str]
) -> List[ImageEmbeddingResult]:
    embedding_model = YOLO(model_path)
    embeddings = []

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image at path {image_path}. Skipping.")
            continue

        embedding = embedding_model.embed(image, verbose=False)[0].cpu().numpy()
        embeddings.append(
            ImageEmbeddingResult(
                image_path=image_path,
                embedding=embedding,
            )
        )

    return embeddings


def compute_embeddings_mp(
    model_path: str, image_paths: List[str], num_processes: int = 6
) -> List[ImageEmbeddingResult]:
    if not image_paths:
        return []

    chunk_size = max(1, len(image_paths) // num_processes)
    chunks = []
    for i in range(0, len(image_paths), chunk_size):
        chunks.append(image_paths[i : i + chunk_size])

    with multiprocessing.Pool(processes=num_processes) as pool:
        worker_func = partial(_compute_embeddings, model_path)
        results = pool.map(worker_func, chunks)

    flattened_results = [item for sublist in results for item in sublist]
    return flattened_results
