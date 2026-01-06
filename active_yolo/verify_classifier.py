import os
import glob
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import List, Optional, Tuple
from PIL import Image, ImageTk
import torch
from ultralytics import YOLO

from config import AppConfig, DataConfig
from label import Label


class CropData:
    """Represents a single cropped bounding box"""
    def __init__(
        self,
        image_path: str,
        crop_image: Image.Image,
        ground_truth_class: int,
        bbox_index: int,
        bbox_coords: Tuple[int, int, int, int],
    ):
        self.image_path = image_path
        self.crop_image = crop_image
        self.ground_truth_class = ground_truth_class
        self.bbox_index = bbox_index
        self.bbox_coords = bbox_coords  # (x1, y1, x2, y2)
        self.predicted_class: Optional[int] = None
        self.prediction_confidence: Optional[float] = None


class ClassifierVerifier:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Classifier Verifier")
        self.root.geometry("1000x700")

        self.app_config = AppConfig.load_app_config()
        self.data_config = DataConfig.load_data_config()

        self.crops: List[CropData] = []
        self.current_index = 0
        self.current_split = "train"  # "train" or "val"
        self.classifier_model = None
        self.model_path: Optional[str] = None
        self.padding_percent = 10  # 10% padding on all sides

        self.display_image: Optional[ImageTk.PhotoImage] = None

        self._setup_ui()
        self._setup_keybindings()

    def _setup_ui(self) -> None:
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Classifier Model", command=self._load_model)
        file_menu.add_command(label="Reload Dataset", command=self._load_dataset)

        # Toolbar
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Button(toolbar, text="Previous", command=self._previous_crop).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(toolbar, text="Next", command=self._next_crop).pack(
            side=tk.LEFT, padx=2
        )

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Split selection
        ttk.Label(toolbar, text="Split:").pack(side=tk.LEFT, padx=5)
        self.split_var = tk.StringVar(value="train")
        train_radio = ttk.Radiobutton(
            toolbar, text="Train", variable=self.split_var, value="train", command=self._switch_split
        )
        train_radio.pack(side=tk.LEFT, padx=2)
        val_radio = ttk.Radiobutton(
            toolbar, text="Val", variable=self.split_var, value="val", command=self._switch_split
        )
        val_radio.pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Padding control
        ttk.Label(toolbar, text="Padding %:").pack(side=tk.LEFT, padx=5)
        self.padding_var = tk.StringVar(value="10")
        padding_spinbox = ttk.Spinbox(
            toolbar, from_=0, to=50, textvariable=self.padding_var, width=5,
            command=self._on_padding_change
        )
        padding_spinbox.pack(side=tk.LEFT, padx=2)

        # Main content
        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - image display
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(left_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Right panel - info
        right_frame = ttk.Frame(content_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)

        # Crop counter
        self.counter_label = ttk.Label(right_frame, text="0/0", font=("Arial", 14, "bold"))
        self.counter_label.pack(anchor=tk.W, pady=(0, 10))

        # Image info
        ttk.Label(right_frame, text="Source Image:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 5))
        self.image_name_label = ttk.Label(right_frame, text="", wraplength=280)
        self.image_name_label.pack(anchor=tk.W)

        # Bounding box info
        ttk.Label(right_frame, text="Bounding Box:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 5))
        self.bbox_info_label = ttk.Label(right_frame, text="", wraplength=280)
        self.bbox_info_label.pack(anchor=tk.W)

        # Ground truth
        ttk.Label(right_frame, text="Ground Truth:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 5))
        self.gt_label = ttk.Label(right_frame, text="", font=("Arial", 12), foreground="blue")
        self.gt_label.pack(anchor=tk.W)

        # Prediction
        ttk.Label(right_frame, text="Prediction:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 5))
        self.pred_label = ttk.Label(right_frame, text="", font=("Arial", 12))
        self.pred_label.pack(anchor=tk.W)

        # Confidence
        self.confidence_label = ttk.Label(right_frame, text="", font=("Arial", 10))
        self.confidence_label.pack(anchor=tk.W, pady=(5, 0))

        # Match indicator
        self.match_label = ttk.Label(right_frame, text="", font=("Arial", 12, "bold"))
        self.match_label.pack(anchor=tk.W, pady=(10, 0))

        # Statistics section
        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(right_frame, text="Statistics:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(5, 5))
        self.stats_label = ttk.Label(right_frame, text="", wraplength=280, justify=tk.LEFT)
        self.stats_label.pack(anchor=tk.W)

        # Status bar
        self.status_label = ttk.Label(self.root, text="Load a classifier model to begin", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def _setup_keybindings(self) -> None:
        self.root.bind("<Left>", lambda e: self._previous_crop())
        self.root.bind("<Right>", lambda e: self._next_crop())
        self.root.bind("<Key-t>", lambda e: self._switch_to_split("train"))
        self.root.bind("<Key-v>", lambda e: self._switch_to_split("val"))

    def _load_model(self) -> None:
        """Load classifier model"""
        model_path = filedialog.askopenfilename(
            title="Select Classifier Model",
            filetypes=[
                ("PyTorch Models", "*.pt *.pth"),
                ("All Files", "*.*")
            ]
        )

        if not model_path:
            return

        try:
            # Try loading as YOLO classifier
            self.classifier_model = YOLO(model_path)
            self.model_path = model_path
            self.status_label.config(text=f"Model loaded: {os.path.basename(model_path)}")
            
            # Load dataset and run predictions
            self._load_dataset()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")

    def _load_dataset(self) -> None:
        """Load dataset and extract crops"""
        if not self.classifier_model:
            messagebox.showwarning("Warning", "Please load a classifier model first")
            return

        self.status_label.config(text="Loading dataset...")
        self.root.update()

        try:
            self.crops = []
            dataset_path = self.app_config.dataset_path
            
            # Load both splits
            for split in ["train", "val"]:
                split_crops = self._load_split(split)
                self.crops.extend(split_crops)
            
            if self.crops:
                # Run predictions on all crops
                self._run_predictions()
                
                # Filter to current split and show first crop
                self.current_index = 0
                self._filter_crops_by_split()
                self._update_display()
                self._update_statistics()
                self.status_label.config(text=f"Loaded {len(self.crops)} crops")
            else:
                self.status_label.config(text="No crops found in dataset")
                messagebox.showinfo("Info", "No labeled images found in dataset")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")
            self.status_label.config(text="Error loading dataset")

    def _load_split(self, split: str) -> List[CropData]:
        """Load crops from a specific split"""
        crops = []
        dataset_path = self.app_config.dataset_path
        images_dir = os.path.join(dataset_path, "images", split)
        labels_dir = os.path.join(dataset_path, "labels", split)

        if not os.path.exists(images_dir):
            print(f"Warning: {images_dir} does not exist")
            return crops

        # Get all images
        image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
        
        for image_path in image_files:
            # Get corresponding label file
            image_basename = os.path.basename(image_path)
            label_filename = image_basename.replace(".jpg", ".txt")
            label_path = os.path.join(labels_dir, label_filename)

            if not os.path.exists(label_path):
                continue

            try:
                # Load image
                image = Image.open(image_path)
                img_width, img_height = image.size

                # Parse labels
                label = Label.parse_label_file(label_path)

                # Extract each bounding box as a crop
                for idx, annotation in enumerate(label.annotations):
                    # Convert YOLO format to pixel coordinates
                    x_center = annotation.x_center * img_width
                    y_center = annotation.y_center * img_height
                    width = annotation.width * img_width
                    height = annotation.height * img_height

                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)

                    # Add padding
                    padding_x = int(width * self.padding_percent / 100)
                    padding_y = int(height * self.padding_percent / 100)

                    x1_padded = max(0, x1 - padding_x)
                    y1_padded = max(0, y1 - padding_y)
                    x2_padded = min(img_width, x2 + padding_x)
                    y2_padded = min(img_height, y2 + padding_y)

                    # Crop image
                    crop = image.crop((x1_padded, y1_padded, x2_padded, y2_padded))

                    # Create crop data
                    crop_data = CropData(
                        image_path=image_path,
                        crop_image=crop,
                        ground_truth_class=annotation.id,
                        bbox_index=idx,
                        bbox_coords=(x1, y1, x2, y2)
                    )
                    # Store split info
                    crop_data.split = split
                    crops.append(crop_data)

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

        return crops

    def _run_predictions(self) -> None:
        """Run classifier predictions on all crops"""
        self.status_label.config(text="Running predictions...")
        self.root.update()

        total = len(self.crops)
        for i, crop_data in enumerate(self.crops):
            if i % 10 == 0:
                self.status_label.config(text=f"Predicting {i}/{total}...")
                self.root.update()

            try:
                # Run prediction
                results = self.classifier_model.predict(
                    crop_data.crop_image,
                    verbose=False
                )

                if results and len(results) > 0:
                    # Get top prediction
                    probs = results[0].probs
                    if probs is not None:
                        top_class = int(probs.top1)
                        top_conf = float(probs.top1conf)
                        
                        crop_data.predicted_class = top_class
                        crop_data.prediction_confidence = top_conf

            except Exception as e:
                print(f"Error predicting crop {i}: {e}")
                continue

        self.status_label.config(text=f"Predictions complete")

    def _filter_crops_by_split(self) -> None:
        """Filter crops to only show current split"""
        self.filtered_crops = [
            crop for crop in self.crops 
            if hasattr(crop, 'split') and crop.split == self.current_split
        ]
        
        # Reset index if out of bounds
        if self.current_index >= len(self.filtered_crops):
            self.current_index = 0

    def _switch_split(self) -> None:
        """Switch between train and val splits"""
        self.current_split = self.split_var.get()
        self._filter_crops_by_split()
        self.current_index = 0
        self._update_display()
        self._update_statistics()

    def _switch_to_split(self, split: str) -> None:
        """Switch to specific split via keyboard"""
        self.split_var.set(split)
        self._switch_split()

    def _on_padding_change(self) -> None:
        """Handle padding change"""
        try:
            new_padding = int(self.padding_var.get())
            if 0 <= new_padding <= 50:
                self.padding_percent = new_padding
                messagebox.showinfo("Info", "Padding changed. Reload dataset to apply.")
        except ValueError:
            pass

    def _previous_crop(self) -> None:
        """Navigate to previous crop"""
        if not self.filtered_crops:
            return
        
        if self.current_index > 0:
            self.current_index -= 1
            self._update_display()

    def _next_crop(self) -> None:
        """Navigate to next crop"""
        if not self.filtered_crops:
            return
        
        if self.current_index < len(self.filtered_crops) - 1:
            self.current_index += 1
            self._update_display()

    def _update_display(self) -> None:
        """Update display with current crop"""
        if not self.filtered_crops or self.current_index >= len(self.filtered_crops):
            self.canvas.delete("all")
            return

        crop_data = self.filtered_crops[self.current_index]

        # Update counter
        self.counter_label.config(
            text=f"{self.current_index + 1}/{len(self.filtered_crops)} ({self.current_split})"
        )

        # Display crop image
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            self.root.after(50, self._update_display)
            return

        # Scale image to fit canvas
        img = crop_data.crop_image
        img_width, img_height = img.size
        
        scale_x = canvas_width / img_width if img_width > 0 else 1
        scale_y = canvas_height / img_height if img_height > 0 else 1
        scale = min(scale_x, scale_y, 1.0)  # Don't scale up

        if scale < 1.0:
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        self.display_image = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2, 
            canvas_height // 2, 
            anchor=tk.CENTER, 
            image=self.display_image
        )

        # Update info labels
        self.image_name_label.config(text=os.path.basename(crop_data.image_path))
        
        bbox_text = f"Box {crop_data.bbox_index + 1}\n"
        bbox_text += f"Coords: ({crop_data.bbox_coords[0]}, {crop_data.bbox_coords[1]}) - "
        bbox_text += f"({crop_data.bbox_coords[2]}, {crop_data.bbox_coords[3]})"
        self.bbox_info_label.config(text=bbox_text)

        # Ground truth
        gt_class_name = self.data_config.names.get(
            crop_data.ground_truth_class, 
            str(crop_data.ground_truth_class)
        )
        self.gt_label.config(text=f"{crop_data.ground_truth_class}: {gt_class_name}")

        # Prediction
        if crop_data.predicted_class is not None:
            pred_class_name = self.data_config.names.get(
                crop_data.predicted_class,
                str(crop_data.predicted_class)
            )
            self.pred_label.config(text=f"{crop_data.predicted_class}: {pred_class_name}")
            
            if crop_data.prediction_confidence is not None:
                self.confidence_label.config(
                    text=f"Confidence: {crop_data.prediction_confidence:.2%}"
                )

            # Match indicator
            if crop_data.predicted_class == crop_data.ground_truth_class:
                self.match_label.config(text="✓ MATCH", foreground="green")
                self.pred_label.config(foreground="green")
            else:
                self.match_label.config(text="✗ MISMATCH", foreground="red")
                self.pred_label.config(foreground="red")
        else:
            self.pred_label.config(text="No prediction")
            self.confidence_label.config(text="")
            self.match_label.config(text="")

    def _update_statistics(self) -> None:
        """Calculate and display accuracy statistics"""
        if not self.filtered_crops:
            self.stats_label.config(text="No data")
            return

        total = len(self.filtered_crops)
        correct = sum(
            1 for crop in self.filtered_crops
            if crop.predicted_class == crop.ground_truth_class
        )
        
        accuracy = (correct / total * 100) if total > 0 else 0
        
        # Per-class stats
        class_stats = {}
        for crop in self.filtered_crops:
            gt = crop.ground_truth_class
            if gt not in class_stats:
                class_stats[gt] = {"total": 0, "correct": 0}
            
            class_stats[gt]["total"] += 1
            if crop.predicted_class == gt:
                class_stats[gt]["correct"] += 1

        stats_text = f"Overall Accuracy: {accuracy:.1f}% ({correct}/{total})\n\n"
        stats_text += "Per-Class Accuracy:\n"
        
        for class_id in sorted(class_stats.keys()):
            class_name = self.data_config.names.get(class_id, str(class_id))
            stats = class_stats[class_id]
            class_acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
            stats_text += f"{class_id}: {class_name}\n"
            stats_text += f"  {class_acc:.1f}% ({stats['correct']}/{stats['total']})\n"

        self.stats_label.config(text=stats_text)

    def run(self) -> None:
        """Start the application"""
        self.root.mainloop()


def main():
    app = ClassifierVerifier()
    app.run()


if __name__ == "__main__":
    main()
