import glob
import os
import shutil
from typing import Counter, Dict, List

from config import AppConfig, DataConfig
from label import Label
from tqdm import tqdm

from dataset import random_split, stratified_split  # type: ignore[unresolved-import]


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
    labels: List[Label], split: str, dataset_path: str, images_path: str
) -> None:
    for label in tqdm(labels, desc=f"Copying {split} files", unit="files"):
        label_filename = os.path.basename(label.file_path)
        image_filename = label_filename.replace(".txt", ".jpg")

        dest_label_path = os.path.join(dataset_path, "labels", split, label_filename)
        dest_image_path = os.path.join(dataset_path, "images", split, image_filename)

        shutil.copy2(label.file_path, dest_label_path)
        shutil.copy2(os.path.join(images_path, image_filename), dest_image_path)


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
    data_config = DataConfig.load_data_config()
    app_config = AppConfig.load_app_config()

    # Delete the current dataset if it exists
    try:
        shutil.rmtree(app_config.dataset_path)
    except FileNotFoundError:
        print(f"No directory exists at {app_config.dataset_path}, continuing")

    _create_directory_structure(app_config.dataset_path)

    labels = _collect_all_labels(app_config.labels_path)

    print(f"Found {len(labels)} labels in total.")

    labels_with_data = [label for label in labels if not label.is_empty()]
    print(f"{len(labels_with_data)} labels contain objects.")

    train_labels, val_labels = (
        stratified_split(labels, app_config.dataset.val_split)
        if app_config.dataset.balance_classes
        else random_split(labels, app_config.dataset.val_split)
    )

    print(
        f"Training has {len(train_labels)} labels and validation has {len(val_labels)} labels."
    )

    _copy_files(
        train_labels, "train", app_config.dataset_path, app_config.raw_images_path
    )
    _copy_files(val_labels, "val", app_config.dataset_path, app_config.raw_images_path)

    def get_class_distribution(labels: List[Label]) -> None:
        count = Counter()  # type: ignore[call-non-callable]
        for label in labels:
            ids = label.get_class_ids()
            count.update(ids)

        print("Class distribution:")
        for class_id, class_count in sorted(count.items()):
            print(f"{class_id}:  {class_count}")

    print("Training set class distribution:")
    get_class_distribution(train_labels)

    print("Validation set class distribution:")
    get_class_distribution(val_labels)

    _create_dataset_yaml(app_config.dataset_path, data_config.names)


if __name__ == "__main__":
    generate_dataset()
