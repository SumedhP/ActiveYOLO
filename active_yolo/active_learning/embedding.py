from functools import partial
from ultralytics.engine.results import Results
import numpy as np
from ultralytics import YOLO  # type: ignore[reportPrivateImportUsage]
from typing import List
from dataclasses import dataclass
import cv2
import multiprocessing


def compute_entropy(model_output: Results) -> float:
    BACKGROUND_ENTROPY = 0.0

    if (
        model_output.boxes is None
        or len(model_output.boxes) == 0
        or model_output.boxes.conf is None
    ):
        return BACKGROUND_ENTROPY

    confidences = model_output.boxes.conf.cpu().numpy()  # type: ignore[possibly-missing-attribute]

    eps = 1e-6
    p = np.clip(confidences, eps, 1 - eps)
    # Bernoulli mean entropy
    entropy = p * -np.log(p) + (1 - p) * -np.log(1 - p)
    return float(np.mean(entropy))


@dataclass
class ImageEmbeddingResult:
    image_path: str
    embedding: np.ndarray
    entropy: float


def _compute_embeddings_and_entropy(
    model_path: str, image_paths: List[str]
) -> List[ImageEmbeddingResult]:
    prediction_model = YOLO(model_path)
    embeddding_model = YOLO(model_path)
    results = []

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image at path {image_path}. Skipping.")
            continue

        embedding = embeddding_model.embed(image, verbose=False)[0].cpu().numpy()
        prediction_results = prediction_model.predict(image, conf=1e-3, verbose=False)[
            0
        ].cpu()
        entropy = compute_entropy(prediction_results)

        results.append(
            ImageEmbeddingResult(
                image_path=image_path,
                embedding=embedding,
                entropy=entropy,
            )
        )

    return results


def compute_embeddings_and_entropy_mp(
    model_path: str, image_paths: List[str], num_processes: int = 6
) -> List[ImageEmbeddingResult]:
    if not image_paths:
        return []

    chunk_size = max(1, len(image_paths) // num_processes)
    chunks = []
    for i in range(0, len(image_paths), chunk_size):
        chunks.append(image_paths[i : i + chunk_size])

    with multiprocessing.Pool(processes=num_processes) as pool:
        worker_func = partial(_compute_embeddings_and_entropy, model_path)
        results = pool.map(worker_func, chunks)

    flattened_results = [item for sublist in results for item in sublist]
    return flattened_results


def _compute_embeddings(model_path: str, image_paths: List[str]) -> List[np.ndarray]:
    embeddding_model = YOLO(model_path)
    embeddings = []

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image at path {image_path}. Skipping.")
            continue

        embedding = embeddding_model.embed(image, verbose=False)[0].cpu().numpy()
        embeddings.append(embedding)

    return embeddings


def compute_embeddings_mp(
    model_path: str, image_paths: List[str], num_processes: int = 6
) -> List[np.ndarray]:
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
