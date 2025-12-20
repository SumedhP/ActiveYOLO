from time import sleep
from typing import List
import os
import glob
from ultralytics import YOLO  # type: ignore[reportPrivateImportUsage]
from config import AppConfig
from active_learning import (
    ImageData,
    compute_entropies_mp,
    compute_embeddings_mp,
    cluster_images,
)


def export_images(image_data_list: List[ImageData], export_file_path: str) -> None:
    os.makedirs(os.path.dirname(export_file_path), exist_ok=True)
    with open(export_file_path, "w") as file:
        for image_data in image_data_list:
            file.write(f"{image_data.image_path} {image_data.entropy}\n")


def compute_low_confidence_images():
    app_config = AppConfig.load_app_config()
    active_learning_config = app_config.active_learning
    print("Loaded app config")

    image_pattern = os.path.join(app_config.raw_images_path, "*.jpg")
    all_images = glob.glob(image_pattern)

    # Filter out images that already have labels
    low_confidence_images = []
    for image_path in all_images:
        # Get corresponding label filename
        image_filename = os.path.basename(image_path)
        label_filename = image_filename.replace(".jpg", ".txt")
        label_path = os.path.join(app_config.labels_path, label_filename)

        # Only include images without existing labels
        if not os.path.exists(label_path):
            low_confidence_images.append(image_path)

    print(f"Total images: {len(all_images)}")
    print(f"Unlabeled images: {len(low_confidence_images)}")

    if not low_confidence_images:
        print("No unlabeled images found. All images have been labeled!")
        return

    model_path = app_config.active_learning.model
    try:
        model = YOLO(model_path)
        model_path = model.export(format="engine", nms=True)
        del model
        sleep(5)  # Time to cleanup nvidia resources
    except Exception as e:
        print(
            f"Error exporting model to TensorRT: {e}, continuing with the current model"
        )

    cpu_count = os.cpu_count()
    if cpu_count is not None:
        num_processes = min(32, cpu_count + 4)
    else:
        num_processes = 4

    entropies = compute_entropies_mp(
        model_path, low_confidence_images, num_processes=num_processes
    )
    embeddings = compute_embeddings_mp(
        model_path, low_confidence_images, num_processes=num_processes
    )

    image_data_list: List[ImageData] = []
    for image_pattern in low_confidence_images:
        entropy = entropies[image_pattern]
        embedding = embeddings[image_pattern]
        image_data_list.append(ImageData(image_pattern, entropy, embedding))

    image_data_list = sorted(image_data_list, key=lambda x: x.entropy, reverse=True)
    print("Top 10 images by entropy:")
    for data in image_data_list[:10]:
        print(f"{data.image_path} {data.entropy}")

    selected_images = cluster_images(
        image_data_list,
        num_clusters=active_learning_config.num_clusters,
        num_images=active_learning_config.images_per_iteration,
    )

    print("Top 10 images post clustering:")
    for data in selected_images[:10]:
        print(f"{data.image_path} {data.entropy}")

    output_file = os.path.join(
        app_config.output_path, active_learning_config.output_file_name
    )
    export_images(selected_images, output_file)


if __name__ == "__main__":
    compute_low_confidence_images()
