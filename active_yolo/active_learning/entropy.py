import numpy as np
from ultralytics import YOLO  # type: ignore[reportPrivateImportUsage]
from ultralytics.engine.results import Results
from typing import List, Dict
from functools import partial
from tqdm.contrib.concurrent import process_map


def compute_entropy(model: YOLO, image_path: str) -> float:
    results = model.predict(image_path, conf=1e-3, verbose=False)

    actual_result = None
    for result in results:
        if isinstance(result, Results):
            actual_result = result
            break

    if actual_result is None:
        print(f"No valid results for image: {image_path}")
        return 0.0

    entropy = 0.0
    if actual_result.boxes is None or len(actual_result.boxes) == 0:
        # Low-priority for images with no detections
        return 1.0

    if actual_result.boxes.conf is None or len(actual_result.boxes.conf) == 0:
        return entropy

    confidences = actual_result.boxes.conf.cpu().numpy()  # type: ignore[possibly-missing-attribute]
    if len(confidences) == 0:
        return entropy

    eps = 1e-10
    p = np.clip(confidences, eps, 1 - eps)
    # Bernoulli mean entropy
    entropy = p * -np.log(p) + (1 - p) * -np.log(1 - p)
    entropy = float(np.mean(entropy))
    return entropy


def _compute_entropy_mp_worker(
    model_path: str, image_paths: List[str]
) -> Dict[str, float]:
    model = YOLO(model_path)
    entropies = {}

    for image_path in image_paths:
        entropy = compute_entropy(model, image_path)
        entropies[image_path] = entropy

    return entropies


def compute_entropies_mp(
    model_path: str, image_paths: List[str], num_processes: int = 6
) -> Dict[str, float]:
    if not image_paths:
        return {}

    chunk_size = max(1, len(image_paths) // num_processes)
    chunks = []
    for i in range(0, len(image_paths), chunk_size):
        chunks.append(image_paths[i : i + chunk_size])

    worker_func = partial(_compute_entropy_mp_worker, model_path)

    results = process_map(
        worker_func,
        chunks,
        max_workers=num_processes,
        desc="Computing entropies",
        unit="chunk",
    )

    final_entropies = {}
    for result in results:
        final_entropies.update(result)

    return final_entropies
