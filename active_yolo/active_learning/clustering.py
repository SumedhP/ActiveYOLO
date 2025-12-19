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

    num_clusters = min(num_clusters, len(image_data))

    embeddings = np.array([data.embedding for data in image_data])
    embeddings = normalize(embeddings, axis=1, norm="l2")
    cluster_labels = KMeans(n_clusters=num_clusters).fit_predict(embeddings)

    selected_images: Set[ImageData] = set()
    for i in range(num_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_metrics = [image_data[j] for j in cluster_indices]

        if len(cluster_metrics) > 0:
            cluster_metrics.sort(key=lambda x: x.entropy, reverse=True)
            selected_images.add(cluster_metrics[0])

    # If we still need more images, add them based on entropy
    if len(selected_images) < num_images:
        remaining_images = [data for data in image_data if data not in selected_images]
        remaining_images.sort(key=lambda x: x.entropy, reverse=True)
        for i in range(num_images - len(selected_images)):
            selected_images.add(remaining_images[i])

    output = list(selected_images)
    output.sort(key=lambda x: x.entropy, reverse=True)

    return output
