import numpy as np
from ultralytics import YOLO  # type: ignore[reportPrivateImportUsage]
from typing import List, Dict
from functools import partial
from tqdm.contrib.concurrent import process_map


def compute_embedding(model: YOLO, image_path: str) -> np.ndarray:
    embedding = model.embed(image_path, verbose=False)[0].numpy()
    return embedding


def _compute_embedding_mp_worker(
    model_path: str, image_paths: List[str]
) -> Dict[str, np.ndarray]:
    model = YOLO(model_path)
    embeddings = {}

    for image_path in image_paths:
        embedding = model.embed(image_path, verbose=False)[0].numpy()
        embeddings[image_path] = embedding

    return embeddings


def compute_embeddings_mp(
    model_path: str, image_paths: List[str], num_processes: int = 6
) -> Dict[str, np.ndarray]:
    if not image_paths:
        return {}

    chunk_size = max(1, len(image_paths) // num_processes)
    chunks = []
    for i in range(0, len(image_paths), chunk_size):
        chunks.append(image_paths[i : i + chunk_size])

    worker_func = partial(_compute_embedding_mp_worker, model_path)

    results = process_map(
        worker_func,
        chunks,
        max_workers=num_processes,
        desc="Computing embeddings",
        unit="chunk",
    )

    final_embeddings = {}
    for result in results:
        final_embeddings.update(result)

    return final_embeddings
