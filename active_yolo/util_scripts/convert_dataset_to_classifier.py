import glob
import os
import sys

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import AppConfig, DataConfig
from label import Label


def crop_object_with_padding(
    image: Image.Image,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    padding_percent: float = 0.1,
) -> Image.Image:
    """
    Crop an object from an image with padding on all sides.

    Args:
        image: PIL Image
        x_center, y_center, width, height: YOLO format (normalized 0-1)
        padding_percent: Percentage of image dimensions to add as padding

    Returns:
        Cropped PIL Image
    """
    img_width, img_height = image.size

    # Convert normalized coordinates to pixel coordinates
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width * (1.0 + padding_percent)
    height_px = height * img_height * (1.0 + padding_percent)

    # Calculate bounding box corners
    x1 = x_center_px - width_px / 2
    y1 = y_center_px - height_px / 2
    x2 = x_center_px + width_px / 2
    y2 = y_center_px + height_px / 2

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)

    # Crop the image
    cropped = image.crop((int(x1), int(y1), int(x2), int(y2)))

    # If the min side of the cropped image is less than 32, resize to have min side 32
    min_side = min(cropped.size)
    if min_side < 32:
        scale_factor = 32 / min_side
        new_size = (
            int(cropped.size[0] * scale_factor),
            int(cropped.size[1] * scale_factor),
        )
        cropped = cropped.resize(new_size, Image.LANCZOS)

    return cropped


def process_split(
    split: str,
    dataset_path: str,
    output_path: str,
    class_names: dict,
) -> None:
    """Process train or val split and organize by class folders."""
    labels_path = os.path.join(dataset_path, "labels", split)
    images_path = os.path.join(dataset_path, "images", split)

    # Create split folder (Train or Val with capital letter)
    split_folder_name = split.capitalize()
    output_split_path = os.path.join(output_path, split_folder_name)

    # Create class folders
    for class_id, class_name in class_names.items():
        class_folder = os.path.join(output_split_path, class_name)
        os.makedirs(class_folder, exist_ok=True)

    label_files = glob.glob(os.path.join(labels_path, "*.txt"))

    crop_count = 0
    class_counts = {class_name: 0 for class_name in class_names.values()}

    for label_file in tqdm(
        label_files, desc=f"Processing {split_folder_name} split", unit="files"
    ):
        label = Label.parse_label_file(label_file)

        if label.is_empty():
            continue

        # Get corresponding image
        image_filename = os.path.basename(label_file).replace(".txt", ".jpg")
        image_path = os.path.join(images_path, image_filename)

        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        image = Image.open(image_path)

        # Process each annotation
        for idx, annotation in enumerate(label.annotations):
            cropped_image = crop_object_with_padding(
                image,
                annotation.x_center,
                annotation.y_center,
                annotation.width,
                annotation.height,
            )

            # Create unique filename for this crop
            base_name = os.path.splitext(image_filename)[0]
            crop_filename = f"{base_name}_crop{idx}.jpg"

            # Get class name and save to appropriate folder
            class_name = class_names[annotation.id]
            class_folder = os.path.join(output_split_path, class_name)
            crop_image_path = os.path.join(class_folder, crop_filename)
            cropped_image.save(crop_image_path)

            crop_count += 1
            class_counts[class_name] += 1

        image.close()

    print(f"\n{split_folder_name} split: Created {crop_count} cropped images")
    print("Class distribution:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count}")


def convert_dataset_to_classifier():
    """Convert detection dataset to classification dataset by cropping labeled objects."""
    app_config = AppConfig.load_app_config()
    data_config = DataConfig.load_data_config()

    # Get absolute path to dataset and create classifier folder at same level
    dataset_abs_path = os.path.abspath(app_config.dataset_path)
    dataset_name = os.path.basename(os.path.normpath(dataset_abs_path))
    dataset_parent = os.path.dirname(dataset_abs_path)
    output_path = os.path.join(dataset_parent, f"classifier_{dataset_name}")

    print(f"Source dataset: {app_config.dataset_path}")
    print(f"Output path: {output_path}")
    print("\nCreating classifier dataset structure:")
    print("  Train/ (with class subfolders)")
    print("  Val/ (with class subfolders)")

    # Delete existing output folder if exists
    if os.path.exists(output_path):
        print(f"\nOutput path already exists. Deleting: {output_path}")
        import shutil

        shutil.rmtree(output_path)

    # Create output directory structure
    os.makedirs(output_path, exist_ok=True)

    # Process train and val splits
    process_split("train", app_config.dataset_path, output_path, data_config.names)
    process_split("val", app_config.dataset_path, output_path, data_config.names)

    print(f"\nClassifier dataset created at: {output_path}")


if __name__ == "__main__":
    convert_dataset_to_classifier()
