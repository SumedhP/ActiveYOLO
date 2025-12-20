import numpy as np
from typing import List, Set
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np
from dataclasses import dataclass


@dataclass
class ImageData:
    image_path: str
    entropy: float
    embedding: np.ndarray

    def __hash__(self):
        return hash(self.image_path)


def cluster_images(
    image_data: List[ImageData], num_clusters: int, num_images: int
) -> List[ImageData]:
    if len(image_data) == 0:
        return []

    if num_images >= len(image_data):
        sorted_data = sorted(image_data, key=lambda x: x.entropy, reverse=True)
        return sorted_data

    num_clusters = min(num_clusters, len(image_data))

    embeddings = np.array([data.embedding for data in image_data])
    embeddings = normalize(embeddings, axis=1, norm="l2")
    cluster_labels = KMeans(n_clusters=num_clusters).fit_predict(embeddings)

    selected_images = []

    # Sort each cluster group based on entropy
    cluster_images = {}
    for idx, label in enumerate(cluster_labels):
        if label not in cluster_images:
            cluster_images[label] = []
        cluster_images[label].append(image_data[idx])

    for label in cluster_images:
        cluster_images[label].sort(key=lambda x: x.entropy, reverse=True)

    current_cluster = 0
    while len(selected_images) < num_images:
        cluster_group = cluster_images[current_cluster]

        # Remove first image in cluster
        if cluster_group:
            selected_images.append(cluster_group.pop(0))

        current_cluster = (current_cluster + 1) % num_clusters

    selected_images.sort(key=lambda x: x.entropy, reverse=True)

    return selected_images
