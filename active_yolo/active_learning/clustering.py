from typing import List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from .embedding import ImageEmbeddingResult


def filter_images_entropy(
    image_data: List[ImageEmbeddingResult], num_clusters: int, num_images: int
) -> List[ImageEmbeddingResult]:
    if len(image_data) == 0:
        return []

    if num_images >= len(image_data):
        return sorted(image_data)

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
        cluster_images[label].sort(reverse=True)

    current_cluster = 0
    while len(selected_images) < num_images:
        cluster_group = cluster_images[current_cluster]

        # Remove first image in cluster
        if cluster_group:
            selected_images.append(cluster_group.pop(0))

        current_cluster = (current_cluster + 1) % num_clusters

    return sorted(selected_images, reverse=True)


def filter_images_centroid(
    image_data: List[ImageEmbeddingResult], num_images: int
) -> List[ImageEmbeddingResult]:
    if len(image_data) == 0:
        return []

    if num_images >= len(image_data):
        return sorted(image_data)

    num_clusters = num_images

    embeddings = np.array([data.embedding for data in image_data])
    embeddings = normalize(embeddings, axis=1, norm="l2")
    cluster_labels = KMeans(n_clusters=num_clusters).fit_predict(embeddings)

    selected_images = []
    # Select the image closest to the centroid in each cluster
    for cluster_idx in range(num_clusters):
        cluster_indices = np.where(cluster_labels == cluster_idx)[0]
        cluster_embeddings = embeddings[cluster_indices]

        centroid = np.mean(cluster_embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        cosine_distances = 1 - np.dot(cluster_embeddings, centroid)
        closest_index = cluster_indices[np.argmin(cosine_distances)]

        selected_images.append(image_data[closest_index])

    return sorted(selected_images, reverse=True)
