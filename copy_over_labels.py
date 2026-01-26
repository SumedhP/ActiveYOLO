import shutil
from pathlib import Path

def copy_labels_for_mislabeled_images():
    """
    Copy label files from labels_from_dataset/labels/ to labels/imageset/
    for images listed in mislabbeled_images.txt
    """
    # Define paths
    mislabeled_file = Path("output/mislabbeled_images.txt")
    source_labels_dir = Path("labels_from_dataset/labels")
    target_labels_dir = Path("labels/imageset")
    
    # Create target directory if it doesn't exist
    target_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the mislabeled images file
    with open(mislabeled_file, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    
    copied_count = 0
    missing_count = 0
    
    # Process each image path
    for image_path in image_paths:
        # Extract the image filename without extension
        image_name = Path(image_path).stem
        
        # Construct the label filename
        label_filename = f"{image_name}.txt"
        
        # Source and target label paths
        source_label = source_labels_dir / label_filename
        target_label = target_labels_dir / label_filename
        
        # Copy the label file if it exists
        if source_label.exists():
            shutil.copy2(source_label, target_label)
            copied_count += 1
            print(f"Copied: {label_filename}")
        else:
            missing_count += 1
            print(f"Missing: {label_filename}")
    
    print(f"\nSummary:")
    print(f"Total images: {len(image_paths)}")
    print(f"Labels copied: {copied_count}")
    print(f"Labels missing: {missing_count}")

if __name__ == "__main__":
    copy_labels_for_mislabeled_images()
