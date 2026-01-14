import os
import glob
import shutil
from tqdm import tqdm

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import AppConfig, DataConfig
from label import Label


def convert_class_id(class_id: int) -> int:
    """
    Convert specific class IDs to general red (0) or blue (1).

    red_unknown (0), red_one (1), red_three (2), red_sentry (3) -> red (0)
    blue_unknown (4), blue_one (5), blue_three (6), blue_sentry (7) -> blue (1)
    """
    if class_id in [0, 1, 2, 3]:  # red_*
        return 0
    elif class_id in [4, 5, 6, 7]:  # blue_*
        return 1
    else:
        raise ValueError(f"Unknown class ID: {class_id}")


def process_split(
    split: str,
    dataset_path: str,
    output_path: str,
) -> None:
    """Process train or val split and convert class IDs."""
    labels_path = os.path.join(dataset_path, "labels", split)
    images_path = os.path.join(dataset_path, "images", split)

    output_labels_path = os.path.join(output_path, "labels", split)
    output_images_path = os.path.join(output_path, "images", split)

    os.makedirs(output_labels_path, exist_ok=True)
    os.makedirs(output_images_path, exist_ok=True)

    label_files = glob.glob(os.path.join(labels_path, "*.txt"))

    red_count = 0
    blue_count = 0

    for label_file in tqdm(label_files, desc=f"Processing {split} split", unit="files"):
        label = Label.parse_label_file(label_file)

        # Get corresponding image filename
        label_filename = os.path.basename(label_file)
        image_filename = label_filename.replace(".txt", ".jpg")
        image_path = os.path.join(images_path, image_filename)

        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        # Copy image
        dest_image_path = os.path.join(output_images_path, image_filename)
        shutil.copy2(image_path, dest_image_path)

        # Create new label file with converted class IDs
        dest_label_path = os.path.join(output_labels_path, label_filename)
        with open(dest_label_path, "w") as f:
            for annotation in label.annotations:
                new_class_id = convert_class_id(annotation.id)
                if new_class_id == 0:
                    red_count += 1
                else:
                    blue_count += 1

                f.write(
                    f"{new_class_id} {annotation.x_center} {annotation.y_center} "
                    f"{annotation.width} {annotation.height}\n"
                )

    print(f"\n{split.capitalize()} split:")
    print(f"  red: {red_count}")
    print(f"  blue: {blue_count}")


def create_dataset_yaml(output_path: str) -> None:
    """Create dataset.yaml with red and blue classes."""
    yaml_content = f"""path: {os.path.abspath(output_path)}
train: images/train
val: images/val

names:
  0: red
  1: blue

nc: 2
"""

    yaml_path = os.path.join(output_path, "dataset.yaml")
    with open(yaml_path, "w") as yaml_file:
        yaml_file.write(yaml_content)


def convert_dataset_to_detector():
    """Convert multi-class detection dataset to binary red/blue detector dataset."""
    app_config = AppConfig.load_app_config()

    # Get absolute path to dataset and create detector folder at same level
    dataset_abs_path = os.path.abspath(app_config.dataset_path)
    dataset_name = os.path.basename(os.path.normpath(dataset_abs_path))
    dataset_parent = os.path.dirname(dataset_abs_path)
    output_path = os.path.join(dataset_parent, f"detector_{dataset_name}")

    print(f"Source dataset: {app_config.dataset_path}")
    print(f"Output path: {output_path}")
    print(f"\nConverting all red_* classes to 'red' and blue_* classes to 'blue'")

    # Delete existing output folder if exists
    if os.path.exists(output_path):
        print(f"\nOutput path already exists. Deleting: {output_path}")
        shutil.rmtree(output_path)

    # Create output directory structure
    os.makedirs(output_path, exist_ok=True)

    # Process train and val splits
    process_split("train", app_config.dataset_path, output_path)
    process_split("val", app_config.dataset_path, output_path)

    # Create dataset.yaml
    create_dataset_yaml(output_path)

    print(f"\nDetector dataset created at: {output_path}")


if __name__ == "__main__":
    convert_dataset_to_detector()
