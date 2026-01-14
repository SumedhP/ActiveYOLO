import argparse
import glob
import os
import shutil
from typing import List

from active_learning import (
    ImageEmbeddingResult,
    compute_embeddings_and_entropy_mp,
    compute_embeddings_mp,
)
from config import AppConfig
from tqdm import tqdm

from active_yolo.active_learning.clustering import (
    filter_images_centroid,
    filter_images_entropy,
)


def _export_image_list(
    image_data_list: List[ImageEmbeddingResult], export_file_path: str
) -> None:
    os.makedirs(os.path.dirname(export_file_path), exist_ok=True)
    with open(export_file_path, "w") as file:
        for image_data in image_data_list:
            file.write(f"{image_data.image_path} {image_data.entropy}\n")


def _find_unlabeled_images(raw_images_path: str, labels_path: str) -> List[str]:
    image_pattern = os.path.join(raw_images_path, "*.jpg")
    all_images = glob.glob(image_pattern)

    # Filter out images that already have labels
    low_confidence_images = []
    for image_path in all_images:
        # Get corresponding label filename
        image_filename = os.path.basename(image_path)
        label_filename = image_filename.replace(".jpg", ".txt")
        label_path = os.path.join(labels_path, label_filename)

        # Only include images without existing labels
        if not os.path.exists(label_path):
            low_confidence_images.append(image_path)

    print(f"Total images: {len(all_images)}")
    print(f"Unlabeled images: {len(low_confidence_images)}")

    if not low_confidence_images:
        print("No unlabeled images found. All images have been labeled!")
        return []

    return low_confidence_images


def compute_low_confidence_images():
    app_config = AppConfig.load_app_config()
    active_learning_config = app_config.active_learning
    print("Loaded app config")

    low_confidence_images = _find_unlabeled_images(
        app_config.raw_images_path, app_config.labels_path
    )
    if not low_confidence_images:
        return

    model_path = app_config.active_learning.model

    image_data_list = compute_embeddings_and_entropy_mp(
        model_path, low_confidence_images
    )

    image_data_list = sorted(image_data_list, key=lambda x: x.entropy, reverse=True)
    print("Top 10 images by entropy:")
    for data in image_data_list[:10]:
        print(f"{data.image_path} {data.entropy}")

    selected_images = filter_images_entropy(
        image_data_list,
        num_clusters=active_learning_config.num_clusters,
        num_images=active_learning_config.images_per_iteration,
    )

    print("Top 10 images post clustering:")
    for data in selected_images[:10]:
        print(f"{data.image_path} {data.entropy}")

    print(
        f"Number of images with Entropy of 1: {sum(1 for data in selected_images if data.entropy == 1.0)}"
    )

    output_file = os.path.join(
        app_config.output_path, active_learning_config.output_file_name
    )
    _export_image_list(selected_images, output_file)


def compute_diverse_imageset():
    app_config = AppConfig.load_app_config()
    active_learning_config = app_config.active_learning
    print("Loaded app config")

    low_confidence_images = _find_unlabeled_images(
        app_config.raw_images_path, app_config.labels_path
    )
    if not low_confidence_images:
        return

    model_path = app_config.active_learning.model
    image_data_list = compute_embeddings_mp(model_path, low_confidence_images)
    selected_images = filter_images_centroid(
        image_data_list,
        num_images=active_learning_config.images_per_iteration,
    )
    # Export images under the input image folder + _small directory

    output_file = app_config.raw_images_path.rstrip("/") + "_small/"
    os.makedirs(output_file, exist_ok=True)
    for data in tqdm(selected_images):
        image_filename = os.path.basename(data.image_path)
        destination_path = os.path.join(output_file, image_filename)
        shutil.copy2(data.image_path, destination_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial",
        action="store_true",
        help="Compute a diverse initial image set instead of low confidence images.",
    )
    args = parser.parse_args()

    if args.initial:
        compute_diverse_imageset()
    else:
        compute_low_confidence_images()
