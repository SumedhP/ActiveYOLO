from tqdm import tqdm
from ultralytics.engine.results import Results
from typing import List
import os
import glob
from ultralytics import YOLO  # type: ignore[reportPrivateImportUsage]
from config import AppConfig
import numpy as np
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


@dataclass
class ImageData:
    image_path: str
    entropy: float
    embedding: np.ndarray


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
        return entropy

    if actual_result.boxes.conf is None or len(actual_result.boxes.conf) == 0:
        return entropy

    confidences = actual_result.boxes.conf.numpy()  # type: ignore[possibly-missing-attribute]
    if len(confidences) == 0:
        return entropy

    eps = 1e-10
    p = np.clip(confidences, eps, 1 - eps)
    # Bernoulli mean entropy
    entropy = p * -np.log(p) + (1 - p) * -np.log(1 - p)
    entropy = float(np.mean(entropy))
    return entropy


def compute_embedding(model: YOLO, image_path: str) -> np.ndarray:
    embedding = model.embed(image_path, verbose=False)[0].numpy()
    return embedding


def cluster_images(
    image_data: List[ImageData], num_clusters: int, num_images: int
) -> List[ImageData]:
    if len(image_data) == 0:
        return []

    embeddings = np.array([data.embedding for data in image_data])
    embeddings = normalize(embeddings, axis=1, norm="l2")
    cluster_labels = KMeans(n_clusters=num_clusters).fit_predict(embeddings)

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

    selected_images.sort(key=lambda x: x.entropy, reverse=True)

    return selected_images


def export_images(image_data_list: List[ImageData], export_file_path: str) -> None:
    os.makedirs(os.path.dirname(export_file_path), exist_ok=True)
    with open(export_file_path, "w") as file:
        for image_data in image_data_list:
            file.write(f"{image_data.image_path} {image_data.entropy}\n")


def compute_low_confidence_images():
    app_config = AppConfig.load_app_config()
    active_learning_config = app_config.active_learning
    print("Loaded app config")

    image_path = os.path.join(app_config.raw_images_path, "*.jpg")
    low_confidence_images = glob.glob(image_path)
    print(len(low_confidence_images))
    print(low_confidence_images[:10])

    model = YOLO(app_config.active_learning.model)
    try:
        exported_model_path = model.export(format="engine", nms=True)
        model = YOLO(exported_model_path)
    except Exception as e:
        print(
            f"Error exporting model to TensorRT: {e}, continuing with the current model"
        )

    # First pass: compute all entropies
    print("Computing entropies for all images...")
    entropies = {}
    for image_path in tqdm(
        low_confidence_images, desc="Computing entropies", unit="image"
    ):
        # for image_path in low_confidence_images:
        entropy = compute_entropy(model, image_path)
        entropies[image_path] = entropy

    # Second pass: compute all embeddings
    print("Computing embeddings for all images...")
    embeddings = {}
    for image_path in tqdm(
        low_confidence_images, desc="Computing embeddings", unit="image"
    ):
        embedding = compute_embedding(model, image_path)
        embeddings[image_path] = embedding

    # Combine results
    image_data_list: List[ImageData] = []
    for image_path in low_confidence_images:
        entropy = entropies[image_path]
        embedding = embeddings[image_path]
        image_data_list.append(ImageData(image_path, entropy, embedding))

    image_data_list = sorted(image_data_list, key=lambda x: x.entropy, reverse=True)
    print("Top 10 images by entropy:")
    for data in image_data_list[:10]:
        print(f"{data.image_path}")

    selected_images = cluster_images(
        image_data_list,
        num_clusters=active_learning_config.num_clusters,
        num_images=active_learning_config.images_per_iteration,
    )

    output_file = os.path.join(
        app_config.output_path, active_learning_config.output_file_name
    )
    export_images(selected_images, output_file)


if __name__ == "__main__":
    compute_low_confidence_images()
