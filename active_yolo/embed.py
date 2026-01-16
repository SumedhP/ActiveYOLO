import csv
import glob
import os

from active_yolo.active_learning.embedding import compute_embeddings_mp
from active_yolo.config.app_config import AppConfig


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

    # Get all images
    image_pattern = os.path.join(app_config.raw_images_path, "*.jpg")
    image_paths = glob.glob(image_pattern)

    if not image_paths:
        print(f"No images found in {app_config.raw_images_path}")
        return

    print(f"Found {len(image_paths)} images. Computing embeddings...")

    # Compute embeddings using existing MP logic
    # Note: process count hardcoded to 6 in original, maybe make configurable later
    results = compute_embeddings_mp(model_path, image_paths)

    # Save to CSV
    output_csv = os.path.join(
        app_config.output_path, app_config.active_learning.embeddings_file_name
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    print(f"Saving embeddings to {output_csv}...")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "embedding_vector"])

        for res in results:
            # Convert numpy array to string representation for CSV storage
            # We join with spaces or commas. Spaces is common for vector strings.
            emb_str = " ".join(map(str, res.embedding.flatten()))
            writer.writerow([res.image_path, emb_str])

    print("Embedding generation complete.")


if __name__ == "__main__":
    generate_embeddings()
