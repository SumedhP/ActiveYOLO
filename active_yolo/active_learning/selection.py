import csv
import os
import random
import numpy as np
from typing import List, Tuple
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from active_yolo.config.app_config import AppConfig


def load_embeddings(csv_path: str) -> Tuple[List[str], np.ndarray]:
    """Loads embeddings from CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Embeddings file not found: {csv_path}")

    paths = []
    embeddings = []

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # Skip header
        for row in reader:
            if len(row) < 2:
                continue
            paths.append(row[0])
            # Convert space-separated string back to numpy array
            vec = np.fromstring(row[1], sep=" ")
            embeddings.append(vec)

    return paths, np.array(embeddings)


def is_labeled(image_path: str, labels_dir: str) -> bool:
    """Checks if a label file exists for the given image."""
    base_name = os.path.basename(image_path)
    file_name = os.path.splitext(base_name)[0]
    label_path = os.path.join(labels_dir, f"{file_name}.txt")
    return os.path.exists(label_path)


def suggest_active_learning_images():
    """
    Implements the Cluster Coverage active learning strategy.
    1. Clusters all images (labeled + unlabeled).
    2. Prioritizes clusters with lower label coverage.
    3. Selects unlabeled images via round-robin from prioritized clusters.
    """
    app_config = AppConfig.load_app_config()

    embeddings_file = os.path.join(
        app_config.output_path, app_config.active_learning.embeddings_file_name
    )

    # 1. Load Data
    print("Loading embeddings...")
    paths, embeddings = load_embeddings(embeddings_file)
    if not paths:
        print("No embeddings found.")
        return

    # 2. Identify Labeled Status
    labeled_mask = []
    for p in paths:
        labeled_mask.append(is_labeled(p, app_config.labels_path))

    labeled_mask = np.array(labeled_mask)
    unlabeled_indices = np.where(~labeled_mask)[0]

    if len(unlabeled_indices) == 0:
        print("All images are already labeled!")
        return

    # 3. Clustering
    print(
        f"Clustering {len(embeddings)} images into {app_config.active_learning.num_clusters} clusters..."
    )
    norm_embeddings = normalize(embeddings, axis=1, norm="l2")
    kmeans = KMeans(n_clusters=app_config.active_learning.num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(norm_embeddings)

    # 4. Compute Coverage
    # cluster_id -> count of labeled images
    cluster_coverage = defaultdict(int)
    # cluster_id -> list of unlabeled image indices
    cluster_unlabeled_pool = defaultdict(list)

    for idx, cluster_id in enumerate(cluster_labels):
        if labeled_mask[idx]:
            cluster_coverage[cluster_id] += 1
        else:
            cluster_unlabeled_pool[cluster_id].append(idx)

    # 5. Prioritize Clusters (Low coverage first)
    # Get all unique cluster IDs (0 to K-1)
    all_clusters = list(range(app_config.active_learning.num_clusters))
    # Sort by coverage count
    sorted_clusters = sorted(all_clusters, key=lambda c: cluster_coverage[c])

    # 6. Select Images
    target_count = app_config.active_learning.images_per_iteration
    selected_indices = []

    print(f"Selecting {target_count} images via Cluster Coverage strategy...")

    while len(selected_indices) < target_count:
        added_in_this_round = False

        for cluster_id in sorted_clusters:
            pool = cluster_unlabeled_pool[cluster_id]
            if pool:
                # Design Doc: "Extracts random unlabeled image"
                # We pop one random image to avoid picking same one twice
                # Efficient: swap random element to end and pop
                rand_idx = random.randint(0, len(pool) - 1)
                pool[rand_idx], pool[-1] = pool[-1], pool[rand_idx]
                selected_idx = pool.pop()

                selected_indices.append(selected_idx)
                added_in_this_round = True

                if len(selected_indices) >= target_count:
                    break

        if not added_in_this_round:
            # Run out of images
            break

    # 7. Write Output
    output_file = os.path.join(
        app_config.output_path, app_config.active_learning.output_file_name
    )

    print(f"Writing {len(selected_indices)} suggested images to {output_file}")
    with open(output_file, "w") as f:
        for idx in selected_indices:
            f.write(f"{paths[idx]}\n")

    print("Done.")


if __name__ == "__main__":
    suggest_active_learning_images()
