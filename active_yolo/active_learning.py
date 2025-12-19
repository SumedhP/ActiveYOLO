from ultralytics.engine.results import Results
from typing import Tuple, List
import os
import glob
from ultralytics import YOLO  # type: ignore[reportPrivateImportUsage]
from config import AppConfig
import numpy as np
from dataclasses import dataclass
from sklearn.cluster import KMeans


@dataclass
class ImageData:
    image_path: str
    entropy: float
    embedding: np.ndarray


def compute_entropy_and_embedding(
    model: YOLO, image_path: str
) -> Tuple[float, np.ndarray]:
    results = model.predict(image_path)

    def compute_entropy(result: Results) -> float:
        entropy = 0.0

        if result.probs is None:
            # Low-priority for images with no detections
            return entropy

        confidences = result.probs.numpy()
        if len(confidences) == 0:
            return entropy

        eps = 1e-10
        confidences = np.clip(confidences, eps, 1 - eps)
        probs = confidences / np.sum(confidences)
        entropy = np.sum(probs * -np.log(probs))
        return entropy

    entropy = sum(compute_entropy(result) for result in results) / len(results)

    embedding = model.embed(image_path)[0].numpy()
    return entropy, embedding


def cluster_images(
    image_data: List[ImageData], num_clusters: int, num_images: int
) -> List[ImageData]:
    if len(image_data) == 0:
        return []

    embeddings = np.array([data.embedding for data in image_data])
    cluster_labels = KMeans(n_clusters=num_clusters, random_state=0).fit_predict(
        embeddings
    )

    selected_images = []
    for i in range(num_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_metrics = [image_data[j] for j in cluster_indices]

        if len(cluster_metrics) > 0:
            cluster_metrics.sort(key=lambda x: x.entropy, reverse=True)
            selected_images.append(cluster_metrics[0])

    # If we still need more images, add them based on entropy
    if len(selected_images) < num_images:
        remaining_images = [data for data in image_data if data not in selected_images]
        remaining_images.sort(key=lambda x: x.entropy, reverse=True)
        selected_images.extend(remaining_images[: num_images - len(selected_images)])

    return selected_images


def compute_low_confidence_images():
    app_config = AppConfig.load_app_config()

    image_path = os.path.join(app_config.raw_images_path, "*.jpg")
    low_confidence_images = glob.glob(image_path)
    print(len(low_confidence_images))
    print(low_confidence_images[:10])

    model = YOLO(app_config.active_learning.model)
    model = model.export(format="engine", nms=True)


if __name__ == "__main__":
    compute_low_confidence_images()
