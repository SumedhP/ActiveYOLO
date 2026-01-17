import glob
import os
import shutil
from typing import Counter, Dict, List

from config import AppConfig, DataConfig
from label import Label
from tqdm import tqdm

def _create_directory_structure(file_path: str) -> None:
    dirs = [
        "images",
        "images/train",
        "images/val",
        "labels",
        "labels/train",
        "labels/val",
    ]

    for dir_path in dirs:
        path = os.path.join(file_path, dir_path)
        os.makedirs(path, exist_ok=True)


def _collect_all_labels(label_folder_path: str) -> List[Label]:
    labels = []
    label_file_paths = glob.glob(os.path.join(label_folder_path, "*.txt"))
    for label_path in tqdm(label_file_paths, desc="Collecting labels", unit="files"):
        label = Label.parse_label_file(label_path)
        labels.append(label)
    return labels


def _copy_files(
    labels: List[Label], split: str, dataset_path: str, source_images_path: str
) -> None:
    """Copy image and label files to the dataset directory."""
    for label in tqdm(labels, desc=f"Copying {split} files", unit="files"):
        label_filename = os.path.basename(label.file_path)
        image_filename = label_filename.replace(".txt", ".jpg")

        dest_label_path = os.path.join(dataset_path, "labels", split, label_filename)
        dest_image_path = os.path.join(dataset_path, "images", split, image_filename)

        shutil.copy2(label.file_path, dest_label_path)
        
        # Image source is in the same directory structure as labels
        # (e.g., labels/imageset/*.txt -> images/imageset/*.jpg)
        source_image_path = os.path.join(source_images_path, image_filename)
        shutil.copy2(source_image_path, dest_image_path)


def _create_dataset_yaml(dataset_path: str, mapping: Dict[int, str]) -> None:
    yaml_content = f"""path: {dataset_path}
train: images/train
val: images/val

names:
"""

    for class_id, class_name in mapping.items():
        yaml_content += f"  {class_id}: {class_name}\n"

    yaml_content += f"\nnc: {len(mapping)}\n"

    yaml_path = os.path.join(dataset_path, "dataset.yaml")
    with open(yaml_path, "w") as yaml_file:
        yaml_file.write(yaml_content)


def generate_dataset():
    """
    Generate YOLO dataset from labeled images.
    - Train split: Labeled images from imageset
    - Val split: All images from validation_set
    """
    data_config = DataConfig.load_data_config()
    app_config = AppConfig.load_app_config()

    # Delete the current dataset if it exists
    try:
        shutil.rmtree(app_config.dataset_path)
    except FileNotFoundError:
        print(f"No directory exists at {app_config.dataset_path}, continuing")

    _create_directory_structure(app_config.dataset_path)

    # Collect labels from imageset (for training)
    print("Collecting labels from imageset...")
    train_labels = _collect_all_labels(app_config.imageset_labels_path)
    print(f"Found {len(train_labels)} labels in imageset.")
    
    train_labels_with_data = [label for label in train_labels if not label.is_empty()]
    print(f"{len(train_labels_with_data)} imageset labels contain objects.")

    # Collect all labels from validation_set (for validation)
    print("Collecting labels from validation_set...")
    val_labels = _collect_all_labels(app_config.validation_labels_path)
    print(f"Found {len(val_labels)} labels in validation_set.")
    
    val_labels_with_data = [label for label in val_labels if not label.is_empty()]
    print(f"{len(val_labels_with_data)} validation_set labels contain objects.")

    print(
        f"Training has {len(train_labels_with_data)} labels and validation has {len(val_labels_with_data)} labels."
    )

    # Copy files
    _copy_files(
        train_labels_with_data, "train", app_config.dataset_path, app_config.imageset_images_path
    )
    _copy_files(
        val_labels_with_data, "val", app_config.dataset_path, app_config.validation_images_path
    )

    def get_class_distribution(labels: List[Label]) -> None:
        count = Counter()  # type: ignore[call-non-callable]
        for label in labels:
            ids = label.get_class_ids()
            count.update(ids)

        print("Class distribution:")
        for class_id, class_count in sorted(count.items()):
            print(f"{class_id}:  {class_count}")

    print("Training set class distribution:")
    get_class_distribution(train_labels_with_data)

    print("Validation set class distribution:")
    get_class_distribution(val_labels_with_data)

    _create_dataset_yaml(app_config.dataset_path, data_config.names)


if __name__ == "__main__":
    generate_dataset()
