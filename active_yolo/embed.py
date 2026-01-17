import csv
import glob
import os
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from active_learning.embedding import compute_embeddings
from config.app_config import AppConfig


def generate_embeddings():
    """
    Generates embeddings for all images in the raw_images directory
    and saves them to a CSV file.
    """
    app_config = AppConfig.load_app_config()

    # Check if model exists
    model_path = app_config.active_learning.model
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Active learning model not found at {model_path}. "
            "Please train a model first or update config."
        )

    # Get all images from imageset (unlabeled pool for active learning)
    image_pattern = os.path.join(app_config.imageset_images_path, "*.jpg")
    image_paths = glob.glob(image_pattern)

    if not image_paths:
        print(f"No images found in {app_config.imageset_images_path}")
        return

    print(f"Found {len(image_paths)} images. Computing embeddings...")

    # Compute embeddings using existing MP logic
    results = compute_embeddings(model_path, image_paths)

    # Perform Clustering ----------------------------------------------------------------
    print(f"Computed embeddings for {len(results)} images. Running K-Means clustering...")
    
    if len(results) > 0:
        embeddings_matrix = np.array([r.embedding.flatten() for r in results])
        
        # Normalize for cosine similarity behavior with Euclidean distance (KMeans)
        norm_embeddings = normalize(embeddings_matrix, axis=1, norm='l2')
        
        num_clusters = app_config.active_learning.num_clusters
        # Handle edge case where we have fewer images than requested clusters
        effective_k = min(num_clusters, len(results))
        
        kmeans = KMeans(n_clusters=effective_k, random_state=42)
        cluster_labels = kmeans.fit_predict(norm_embeddings)
    else:
        cluster_labels = []

    # Save to CSV -----------------------------------------------------------------------
    output_csv = os.path.join(
        app_config.output_path, app_config.active_learning.embeddings_file_name
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    print(f"Saving embeddings and cluster IDs to {output_csv}...")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "cluster_id", "embedding_vector"])

        for i, res in enumerate(results):
            cid = cluster_labels[i] if i < len(cluster_labels) else -1
            
            # Convert numpy array to string representation for CSV storage
            # We join with spaces or commas. Spaces is common for vector strings.
            emb_str = " ".join(map(str, res.embedding.flatten()))
            writer.writerow([res.image_path, cid, emb_str])

    print("Embedding generation complete.")


if __name__ == "__main__":
    generate_embeddings()
