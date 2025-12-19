import numpy as np
from typing import List
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np
from dataclasses import dataclass


@dataclass
class ImageData:
    image_path: str
    entropy: float
    embedding: np.ndarray


def cluster_images(
    image_data: List[ImageData], num_clusters: int, num_images: int
) -> List[ImageData]:
    if len(image_data) == 0:
        return []

    embeddings = np.array([data.embedding for data in image_data])
    embeddings = normalize(embeddings, axis=1, norm="l2")
    cluster_labels = KMeans(n_clusters=num_clusters).fit_predict(embeddings)

    # Organize images by cluster and sort by entropy within each cluster
    cluster_data = {}
    for i in range(num_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_metrics = [image_data[j] for j in cluster_indices]
        if len(cluster_metrics) > 0:
            cluster_metrics.sort(key=lambda x: x.entropy, reverse=True)
            cluster_data[i] = cluster_metrics

    # Round-robin selection from clusters
    selected_images = []
    cluster_pointers = {
        i: 0 for i in cluster_data.keys()
    }  # Track next image to select from each cluster

    while len(selected_images) < num_images and any(
        cluster_pointers[i] < len(cluster_data[i]) for i in cluster_data.keys()
    ):
        for cluster_id in cluster_data.keys():
            if len(selected_images) >= num_images:
                break

            # Check if this cluster still has unselected images
            if cluster_pointers[cluster_id] < len(cluster_data[cluster_id]):
                selected_images.append(
                    cluster_data[cluster_id][cluster_pointers[cluster_id]]
                )
                cluster_pointers[cluster_id] += 1

    # If we still need more images, add remaining ones based on entropy
    if len(selected_images) < num_images:
        remaining_images = [data for data in image_data if data not in selected_images]
        remaining_images.sort(key=lambda x: x.entropy, reverse=True)
        selected_images.extend(remaining_images[: num_images - len(selected_images)])

    selected_images.sort(key=lambda x: x.entropy, reverse=True)

    return selected_images
