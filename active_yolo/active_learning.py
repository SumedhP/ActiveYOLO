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
    low_confidence_images = glob.glob(image_pattern)
    print(len(low_confidence_images))
    print(low_confidence_images[:10])

    model_path = app_config.active_learning.model
    try:
        model = YOLO(model_path)
        model_path = model.export(format="engine", nms=True)
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
