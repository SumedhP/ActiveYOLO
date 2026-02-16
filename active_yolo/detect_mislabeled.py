"""
Script to identify potentially mislabeled images in the dataset.

This script runs inference on all images in the dataset and compares predictions
with ground truth labels. It flags images with:
- High confidence false positives (predicted boxes not matching any ground truth)
- High confidence incorrect labels (predicted class differs from ground truth)

Output is sorted by confidence and saved to output/mislabeled_images.txt
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm
from ultralytics import YOLO

from config import AppConfig


def iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Boxes are in YOLO format: [x_center, y_center, width, height] (normalized 0-1)
    """
    # Convert from center format to corner format
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2
    
    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2
    
    # Calculate intersection area
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0.0


def load_ground_truth_labels(label_path: str) -> List[Tuple[int, List[float]]]:
    """
    Load ground truth labels from YOLO format label file.
    
    Returns list of (class_id, [x_center, y_center, width, height])
    """
    if not os.path.exists(label_path):
        return []
    
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                bbox = [float(x) for x in parts[1:5]]
                labels.append((class_id, bbox))
    return labels


def check_image_for_mislabels(
    image_path: str,
    label_path: str,
    model: YOLO,
    confidence_threshold: float,
    iou_threshold: float = 0.5
) -> List[Tuple[float, str]]:
    """
    Check a single image for potential mislabeling issues.
    
    Returns list of (confidence, reason) tuples for flagged issues.
    """
    issues = []
    
    # Load ground truth
    gt_labels = load_ground_truth_labels(label_path)
    
    # Run inference
    results = model.predict(
        image_path,
        conf=confidence_threshold,
        verbose=False,
        imgsz=640
    )
    
    if len(results) == 0 or results[0].boxes is None:
        return issues
    
    # Get predictions
    pred_boxes = results[0].boxes
    
    for i in range(len(pred_boxes)):
        pred_conf = float(pred_boxes.conf[i])
        pred_cls = int(pred_boxes.cls[i])
        pred_xyxy = pred_boxes.xyxyn[i].cpu().numpy()  # normalized coordinates
        
        # Convert xyxy to xywh (center format)
        x_center = (pred_xyxy[0] + pred_xyxy[2]) / 2
        y_center = (pred_xyxy[1] + pred_xyxy[3]) / 2
        width = pred_xyxy[2] - pred_xyxy[0]
        height = pred_xyxy[3] - pred_xyxy[1]
        pred_bbox = [x_center, y_center, width, height]
        
        # Find best matching ground truth box
        best_iou = 0.0
        best_gt_class = None
        
        for gt_class, gt_bbox in gt_labels:
            curr_iou = iou(pred_bbox, gt_bbox)
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_gt_class = gt_class
        
        # Check for issues
        if best_iou < iou_threshold:
            # High confidence false positive
            issues.append((
                pred_conf,
                f"False positive (class {pred_cls}, conf={pred_conf:.2f})"
            ))
        elif best_gt_class != pred_cls:
            # Wrong class label
            issues.append((
                pred_conf,
                f"Wrong class (predicted {pred_cls}, labeled {best_gt_class}, conf={pred_conf:.2f})"
            ))
    
    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Detect potentially mislabeled images in the dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/detector.pt",
        help="Path to trained YOLO model"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="Confidence threshold for detection (default: 0.4)"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching predictions to ground truth (default: 0.5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/mislabeled_images_analysis.txt",
        help="Output file path"
    )
    
    args = parser.parse_args()
    
    # Load config
    app_config = AppConfig.load_app_config()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)
    
    # Get images from imageset directory
    image_dir = Path(app_config.imageset_images_path)
    label_dir = Path(app_config.imageset_labels_path)
    
    if not image_dir.exists():
        print(f"Error: {image_dir} does not exist!")
        return
    
    # Collect all flagged images
    flagged_images = []

    print(f"\nProcessing {image_dir}...")
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    for idx, image_path in enumerate(tqdm(image_files), 1):
        
        
        # Get corresponding label file
        label_path = label_dir / f"{image_path.stem}.txt"
        
        # Check for mislabels
        issues = check_image_for_mislabels(
            str(image_path),
            str(label_path),
            model,
            args.confidence,
            args.iou_threshold
        )
        
        if issues:
            # Get highest confidence issue for this image
            max_conf = max(issue[0] for issue in issues)
            issue_descriptions = "; ".join(issue[1] for issue in issues)
            
            flagged_images.append((max_conf, image_path, issue_descriptions))
            flagged_pct = (len(flagged_images) * 1.0 / idx) * 100
            print(f"Found new issue, percent flagged: {flagged_pct:.2f}% images")
    
    # Sort by confidence (descending)
    flagged_images.sort(key=lambda x: x[0], reverse=True)
    print(f"\n✓ Found {len(flagged_images)} potentially mislabeled images")
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"# Potentially Mislabeled Images (confidence >= {args.confidence})\n")
        f.write(f"# Format: confidence | filepath | issues\n")
        f.write(f"# Total flagged: {len(flagged_images)}\n\n")
        
        for conf, filepath, issues in flagged_images:
            f.write(f"{conf:.4f} | {filepath} | {issues}\n")
    
    with open("output/mislabbeled_images.txt", 'w') as f:
        for _, filepath, _ in flagged_images:
            f.write(f"{filepath}\n")


    print(f"✓ Results saved to {output_path}")
    
    if flagged_images:
        print(f"\nTop 5 most suspicious images:")
        for conf, filepath, issues in flagged_images[:5]:
            print(f"  {conf:.2f} - {filepath}")
            print(f"         {issues}")


if __name__ == "__main__":
    main()
