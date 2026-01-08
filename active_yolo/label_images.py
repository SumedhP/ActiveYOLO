import time as time

from typing import List, Optional
import os
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
from ultralytics import YOLO  # type: ignore[reportPrivateImportUsage]

from config import AppConfig, DataConfig
from label import Label, BoundingBox
from generate_dataset import generate_dataset
from analyze_images import compute_low_confidence_images
from train import train_model


class LabelingTool:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ActiveYOLO Labeling Tool")
        self.root.geometry("1200x800")

        self.app_config = AppConfig.load_app_config()
        self.data_config = DataConfig.load_data_config()

        self.current_image_path: Optional[str] = None
        self.current_image: Optional[Image.Image] = None
        self.display_image: Optional[ImageTk.PhotoImage] = None
        self.image_files: List[str] = []
        self.current_index = 0
        self.bounding_boxes: List[BoundingBox] = []
        self.active_learning_mode = False
        self.invert_order = False
        self.auto_load_predictions = False
        self.scale_factor = 1.0
        self.zoom_level = 1.0
        self.user_has_zoomed = False  # Track if user has manually adjusted zoom
        self.entropy_values = {}  # Store entropy values for active learning images
        self.current_entropy = None  # Current image's entropy value

        self.drawing_box = False
        self.resizing_box = False
        self.resize_mode = None  # 'nw', 'ne', 'sw', 'se', 'n', 's', 'e', 'w'
        self.start_x = 0
        self.start_y = 0
        self.current_box: Optional[BoundingBox] = None
        self.selected_box: Optional[BoundingBox] = None
        self.temp_box: Optional[BoundingBox] = None  # For drawing preview
        self.current_class_id = 0

        self.model: Optional[YOLO] = None

        self._setup_ui()
        self._load_images()
        self._setup_keybindings()

        if self.image_files:
            self._load_current_image()

    def _setup_ui(self) -> None:
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Directory", command=self._open_directory)
        file_menu.add_separator()
        file_menu.add_command(label="Generate Dataset", command=self._generate_dataset)
        file_menu.add_command(label="Train Model", command=self._train_model)
        file_menu.add_command(
            label="Run Active Learning Analysis", command=self._run_active_learning
        )

        # Toolbar
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Button(toolbar, text="Previous", command=self._previous_image).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(toolbar, text="Next", command=self._next_image).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(toolbar, text="Save", command=self._save_labels).pack(
            side=tk.LEFT, padx=10
        )

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Active learning list toggle
        self.active_learning_var = tk.BooleanVar()
        ttk.Checkbutton(
            toolbar,
            text="Use Active Learning List",
            variable=self.active_learning_var,
            command=self._toggle_active_learning_list,
        ).pack(side=tk.LEFT, padx=5)

        # Invert order checkbox
        self.invert_order_var = tk.BooleanVar()
        ttk.Checkbutton(
            toolbar,
            text="Invert Order",
            variable=self.invert_order_var,
            command=self._toggle_invert_order,
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            toolbar, text="Jump to Unlabeled", command=self._jump_to_unlabeled
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            toolbar, text="Load Model Suggestions", command=self._load_model_suggestions
        ).pack(side=tk.LEFT, padx=5)

        # Auto-load predictions toggle
        self.auto_load_var = tk.BooleanVar()
        ttk.Checkbutton(
            toolbar,
            text="Auto Load Predictions",
            variable=self.auto_load_var,
            command=self._toggle_auto_load_predictions,
        ).pack(side=tk.LEFT, padx=5)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Entropy display (for active learning mode)
        self.entropy_label = ttk.Label(toolbar, text="", foreground="blue")
        self.entropy_label.pack(side=tk.LEFT, padx=5)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Zoom controls
        ttk.Button(toolbar, text="Zoom In", command=lambda: self._zoom_in()).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(toolbar, text="Zoom Out", command=lambda: self._zoom_out()).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(toolbar, text="Fit", command=self._zoom_fit).pack(
            side=tk.LEFT, padx=2
        )

        # Main content
        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - image canvas
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(left_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Canvas scrollbars
        h_scroll = ttk.Scrollbar(
            left_frame, orient=tk.HORIZONTAL, command=self.canvas.xview
        )
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        v_scroll = ttk.Scrollbar(
            left_frame, orient=tk.VERTICAL, command=self.canvas.yview
        )
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        # Right panel - controls
        right_frame = ttk.Frame(content_frame, width=250)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)

        # Class selection
        ttk.Label(right_frame, text="Class Selection:").pack(anchor=tk.W, pady=(0, 5))

        self.class_var = tk.StringVar()
        self.class_combobox = ttk.Combobox(
            right_frame, textvariable=self.class_var, state="readonly"
        )
        self.class_combobox.pack(fill=tk.X, pady=(0, 10))

        # Update class options
        class_options = [f"{id}: {name}" for id, name in self.data_config.names.items()]
        self.class_combobox["values"] = class_options
        if class_options:
            self.class_combobox.current(0)
            self.class_var.trace("w", self._on_class_change)

        # Image info
        ttk.Label(right_frame, text="Image Info:").pack(anchor=tk.W, pady=(10, 5))

        self.image_info_frame = ttk.Frame(right_frame)
        self.image_info_frame.pack(fill=tk.X, pady=(0, 10))

        self.image_counter_label = ttk.Label(self.image_info_frame, text="0/0")
        self.image_counter_label.pack(anchor=tk.W)

        self.image_name_label = ttk.Label(
            self.image_info_frame, text="", wraplength=200
        )
        self.image_name_label.pack(anchor=tk.W)

        # Bounding boxes list
        ttk.Label(right_frame, text="Bounding Boxes:").pack(anchor=tk.W, pady=(10, 5))

        bbox_frame = ttk.Frame(right_frame)
        bbox_frame.pack(fill=tk.BOTH, expand=True)

        self.bbox_listbox = tk.Listbox(bbox_frame)
        self.bbox_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        bbox_scroll = ttk.Scrollbar(
            bbox_frame, orient=tk.VERTICAL, command=self.bbox_listbox.yview
        )
        bbox_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.bbox_listbox.configure(yscrollcommand=bbox_scroll.set)

        # Bbox controls
        bbox_controls = ttk.Frame(right_frame)
        bbox_controls.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(
            bbox_controls, text="Delete Selected", command=self._delete_selected_box
        ).pack(fill=tk.X, pady=2)

        # Canvas bindings
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)  # Linux
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)  # Linux
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Listbox bindings
        self.bbox_listbox.bind("<<ListboxSelect>>", self._on_bbox_select)

    def _setup_keybindings(self) -> None:
        self.root.bind("<Key>", self._on_key_press)
        self.root.focus_set()

        # Number keys for class selection
        for i in range(10):
            self.root.bind(
                f"<Key-{i}>", lambda e, idx=i: self._set_class_by_number(idx)
            )

        # Navigation
        self.root.bind("<Left>", lambda e: self._previous_image())
        self.root.bind("<Right>", lambda e: self._next_image())

        # Actions
        self.root.bind("<Control-s>", lambda e: self._save_labels())
        self.root.bind("<Delete>", lambda e: self._delete_selected_box())
        self.root.bind("<BackSpace>", lambda e: self._delete_selected_box())

    def _load_images(self) -> None:
        if self.active_learning_mode:
            self._load_active_learning_images()
        else:
            pattern = os.path.join(self.app_config.raw_images_path, "*.jpg")
            self.image_files = sorted(glob.glob(pattern))

        self.current_index = 0
        self._update_image_counter()

    def _load_active_learning_images(self) -> None:
        output_file = os.path.join(
            self.app_config.output_path,
            self.app_config.active_learning.output_file_name,
        )

        self.image_files = []
        self.entropy_values = {}
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        image_path = parts[0]
                        entropy_value = float(parts[1])
                        self.image_files.append(image_path)
                        self.entropy_values[image_path] = entropy_value
                    elif parts:
                        self.image_files.append(parts[0])
        
        # Reverse the list if invert order is enabled
        if self.invert_order:
            self.image_files.reverse()

    def _toggle_active_learning_list(self) -> None:
        self.active_learning_mode = self.active_learning_var.get()
        self._load_images()
        if self.image_files:
            self._load_current_image()

    def _toggle_invert_order(self) -> None:
        self.invert_order = self.invert_order_var.get()
        if self.active_learning_mode:
            self._load_images()
            if self.image_files:
                self._load_current_image()

    def _toggle_auto_load_predictions(self) -> None:
        self.auto_load_predictions = self.auto_load_var.get()

    def _jump_to_unlabeled(self) -> None:
        """Jump to the next unlabeled image"""
        if not self.image_files:
            return

        self._save_labels()  # Auto-save current image

        # Search forward from current position
        start_index = self.current_index + 1
        for i in range(start_index, len(self.image_files)):
            if not self._has_labels(self.image_files[i]):
                self.current_index = i
                self._load_current_image()
                self._update_image_counter()
                return

        # If not found forward, search from beginning
        for i in range(0, start_index):
            if not self._has_labels(self.image_files[i]):
                self.current_index = i
                self._load_current_image()
                self._update_image_counter()
                return

        # No unlabeled images found
        messagebox.showinfo("Info", "All images have been labeled!")

    def _has_labels(self, image_path: str) -> bool:
        """Check if an image has an existing label file"""
        label_filename = os.path.basename(image_path).replace(".jpg", ".txt")
        label_path = os.path.join(self.app_config.labels_path, label_filename)
        return os.path.exists(label_path)

    def _load_current_image(self) -> None:
        if not self.image_files or self.current_index >= len(self.image_files):
            return

        self.current_image_path = self.image_files[self.current_index]

        try:
            self.current_image = Image.open(self.current_image_path)

            # Reset zoom to fit unless user has manually adjusted it
            if not self.user_has_zoomed:
                self.zoom_level = 1.0

            self._load_existing_labels()
            
            # Auto-load model predictions if enabled and no labels exist
            if self.auto_load_predictions and not self.bounding_boxes:
                self._load_model_suggestions()
            
            self._update_display()
            self._update_image_info()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def _update_display(self) -> None:
        if not self.current_image:
            return

        # Calculate scale to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Wait for canvas to be properly initialized
        if canvas_width <= 1 or canvas_height <= 1:
            self.root.after(50, self._update_display)
            return

        img_width, img_height = self.current_image.size
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height

        # If user hasn't manually zoomed and zoom_level is 1.0, allow scaling up to fit canvas
        if not self.user_has_zoomed and self.zoom_level == 1.0:
            base_scale = min(scale_x, scale_y)  # Allow scaling up to fit
        else:
            base_scale = min(
                scale_x, scale_y, 1.0
            )  # Limit to original size for manual zoom

        self.scale_factor = base_scale * self.zoom_level

        display_width = int(img_width * self.scale_factor)
        display_height = int(img_height * self.scale_factor)

        # Resize image for display
        display_img = self.current_image.resize(
            (display_width, display_height), Image.Resampling.LANCZOS
        )

        # Draw bounding boxes
        draw_img = display_img.copy()
        draw = ImageDraw.Draw(draw_img)

        # Draw existing boxes
        boxes_to_draw = self.bounding_boxes.copy()
        if self.temp_box:
            boxes_to_draw.append(self.temp_box)

        for bbox in boxes_to_draw:
            x1 = int(bbox.x1 * self.scale_factor)
            y1 = int(bbox.y1 * self.scale_factor)
            x2 = int(bbox.x2 * self.scale_factor)
            y2 = int(bbox.y2 * self.scale_factor)

            # Handle different box types - use getattr with defaults for missing attributes
            is_suggested = getattr(bbox, "suggested", False)
            is_selected = getattr(bbox, "selected", False)
            is_temp = bbox == self.temp_box

            if is_temp:
                color = "blue"  # Temporary box color
                width = 1
            elif is_suggested:
                color = "red"  # Model suggestion color
                width = 3 if is_selected else 2
            else:
                color = "green"  # Manual annotation color
                width = 3 if is_selected else 2

            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

            # Draw class label
            if not is_temp:
                class_name = self.data_config.names.get(
                    bbox.class_id, str(bbox.class_id)
                )
                draw.text((x1, y1 - 15), f"{bbox.class_id}: {class_name}", fill=color)

        self.display_image = ImageTk.PhotoImage(draw_img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        self._update_bbox_list()

    def _load_existing_labels(self) -> None:
        if not self.current_image_path or not self.current_image:
            return

        label_filename = os.path.basename(self.current_image_path).replace(
            ".jpg", ".txt"
        )
        label_path = os.path.join(self.app_config.labels_path, label_filename)

        self.bounding_boxes = []

        if os.path.exists(label_path):
            try:
                label = Label.parse_label_file(label_path)
                img_width, img_height = self.current_image.size

                for annotation in label.annotations:
                    bbox = BoundingBox.from_yolo_annotation(
                        annotation, img_width, img_height
                    )
                    self.bounding_boxes.append(bbox)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load labels: {e}")

    def _get_resize_mode(self, x: int, y: int, bbox: BoundingBox) -> Optional[str]:
        """Determine if click is near edge/corner for resizing"""
        handle_size = 8  # pixels

        # Check corners first
        if abs(x - bbox.x1) <= handle_size and abs(y - bbox.y1) <= handle_size:
            return "nw"
        if abs(x - bbox.x2) <= handle_size and abs(y - bbox.y1) <= handle_size:
            return "ne"
        if abs(x - bbox.x1) <= handle_size and abs(y - bbox.y2) <= handle_size:
            return "sw"
        if abs(x - bbox.x2) <= handle_size and abs(y - bbox.y2) <= handle_size:
            return "se"

        # Check edges
        if abs(y - bbox.y1) <= handle_size and bbox.x1 <= x <= bbox.x2:
            return "n"
        if abs(y - bbox.y2) <= handle_size and bbox.x1 <= x <= bbox.x2:
            return "s"
        if abs(x - bbox.x1) <= handle_size and bbox.y1 <= y <= bbox.y2:
            return "w"
        if abs(x - bbox.x2) <= handle_size and bbox.y1 <= y <= bbox.y2:
            return "e"

        return None

    def _save_labels(self) -> None:
        if not self.current_image_path or not self.current_image:
            return

        label_filename = os.path.basename(self.current_image_path).replace(
            ".jpg", ".txt"
        )
        label_path = os.path.join(self.app_config.labels_path, label_filename)

        # Ensure labels directory exists
        os.makedirs(os.path.dirname(label_path), exist_ok=True)

        try:
            img_width, img_height = self.current_image.size
            annotations = []

            for bbox in self.bounding_boxes:
                annotation = bbox.to_yolo_annotation(img_width, img_height)
                annotations.append(annotation)

            # Write label file
            with open(label_path, "w") as f:
                for annotation in annotations:
                    f.write(
                        f"{annotation.id} {annotation.x_center:.6f} {annotation.y_center:.6f} {annotation.width:.6f} {annotation.height:.6f}\n"
                    )

            print(f"Saved {len(annotations)} annotations to {label_path}")

            # Mark all boxes as accepted (non-suggested) since they've been saved
            for bbox in self.bounding_boxes:
                if hasattr(bbox, "suggested"):
                    bbox.suggested = False
                else:
                    bbox.suggested = False

            # Refresh display to show updated colors
            self._update_display()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save labels: {e}")

    def _on_canvas_click(self, event) -> None:
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        # Convert to original image coordinates
        orig_x = int(x / self.scale_factor)
        orig_y = int(y / self.scale_factor)

        # Check if clicking on existing box
        clicked_box = None
        resize_mode = None

        for bbox in self.bounding_boxes:
            # Use getattr to safely check for selected attribute
            if hasattr(bbox, "selected") and bbox.selected:
                # Check if clicking on resize handle of selected box
                resize_mode = self._get_resize_mode(orig_x, orig_y, bbox)
                if resize_mode:
                    clicked_box = bbox
                    break

            if bbox.contains_point(orig_x, orig_y):
                clicked_box = bbox
                break

        if clicked_box and resize_mode:
            # Start resizing
            self.resizing_box = True
            self.resize_mode = resize_mode
            self.selected_box = clicked_box
            self.start_x = orig_x
            self.start_y = orig_y
        elif clicked_box:
            # Select the box - add selected attribute if it doesn't exist
            for bbox in self.bounding_boxes:
                if hasattr(bbox, "selected"):
                    bbox.selected = False
                else:
                    bbox.selected = False  # Add the attribute

            if hasattr(clicked_box, "selected"):
                clicked_box.selected = True
            else:
                clicked_box.selected = True  # Add the attribute
            self.selected_box = clicked_box
        else:
            # Start drawing new box
            self.drawing_box = True
            self.start_x = orig_x
            self.start_y = orig_y

            # Clear selections
            for bbox in self.bounding_boxes:
                if hasattr(bbox, "selected"):
                    bbox.selected = False
                else:
                    bbox.selected = False
            self.selected_box = None

        self._update_display()

    def _on_canvas_drag(self, event) -> None:
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        orig_x = int(x / self.scale_factor)
        orig_y = int(y / self.scale_factor)

        if self.resizing_box and self.selected_box and self.resize_mode:
            # Resize the selected box
            bbox = self.selected_box

            if self.resize_mode == "nw":
                bbox.x1 = orig_x
                bbox.y1 = orig_y
            elif self.resize_mode == "ne":
                bbox.x2 = orig_x
                bbox.y1 = orig_y
            elif self.resize_mode == "sw":
                bbox.x1 = orig_x
                bbox.y2 = orig_y
            elif self.resize_mode == "se":
                bbox.x2 = orig_x
                bbox.y2 = orig_y
            elif self.resize_mode == "n":
                bbox.y1 = orig_y
            elif self.resize_mode == "s":
                bbox.y2 = orig_y
            elif self.resize_mode == "w":
                bbox.x1 = orig_x
            elif self.resize_mode == "e":
                bbox.x2 = orig_x

            # Ensure box maintains minimum size and correct order
            if bbox.x1 > bbox.x2:
                bbox.x1, bbox.x2 = bbox.x2, bbox.x1
            if bbox.y1 > bbox.y2:
                bbox.y1, bbox.y2 = bbox.y2, bbox.y1

            self._update_display()

        elif self.drawing_box:
            # Create temporary box for preview
            self.temp_box = BoundingBox(
                min(self.start_x, orig_x),
                min(self.start_y, orig_y),
                max(self.start_x, orig_x),
                max(self.start_y, orig_y),
                self.current_class_id,
            )

            self._update_display()

    def _on_canvas_release(self, event) -> None:
        if self.resizing_box:
            self.resizing_box = False
            self.resize_mode = None
            return

        if not self.drawing_box:
            return

        self.drawing_box = False
        self.temp_box = None  # Clear temporary box

        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        orig_x = int(x / self.scale_factor)
        orig_y = int(y / self.scale_factor)

        # Create final box if minimum size
        if abs(orig_x - self.start_x) > 5 and abs(orig_y - self.start_y) > 5:
            new_box = BoundingBox(
                min(self.start_x, orig_x),
                min(self.start_y, orig_y),
                max(self.start_x, orig_x),
                max(self.start_y, orig_y),
                self.current_class_id,
            )
            self.bounding_boxes.append(new_box)

        self._update_display()

    def _on_bbox_select(self, event) -> None:
        selection = self.bbox_listbox.curselection()
        if selection:
            index = selection[0]
            if 0 <= index < len(self.bounding_boxes):
                # Clear all selections
                for bbox in self.bounding_boxes:
                    if hasattr(bbox, "selected"):
                        bbox.selected = False
                    else:
                        bbox.selected = False

                # Select the chosen box
                selected_bbox = self.bounding_boxes[index]
                if hasattr(selected_bbox, "selected"):
                    selected_bbox.selected = True
                else:
                    selected_bbox.selected = True
                self.selected_box = selected_bbox
                self._update_display()

    def _on_key_press(self, event) -> None:
        if event.keysym in ["Left", "Right", "Delete", "BackSpace"]:
            return  # Handled by specific bindings

        # Handle other keys for shortcuts
        pass

    def _on_class_change(self, *args) -> None:
        selection = self.class_var.get()
        if selection:
            class_id = int(selection.split(":")[0])
            self.current_class_id = class_id

            # Also update selected box if there is one
            if self.selected_box and hasattr(self.selected_box, "class_id"):
                self.selected_box.class_id = class_id
                self._update_display()

    def _set_class_by_number(self, number: int) -> None:
        if number in self.data_config.names:
            self.current_class_id = number
            self.class_combobox.current(
                list(self.data_config.names.keys()).index(number)
            )

            # Also update selected box if there is one
            if self.selected_box and hasattr(self.selected_box, "class_id"):
                self.selected_box.class_id = number
                self._update_display()

    def _previous_image(self) -> None:
        if self.current_index > 0:
            self._save_labels()  # Auto-save
            self.current_index -= 1
            self._load_current_image()
            self._update_image_counter()

    def _next_image(self) -> None:
        if self.current_index < len(self.image_files) - 1:
            self._save_labels()  # Auto-save
            self.current_index += 1
            self._load_current_image()
            self._update_image_counter()

    def _delete_selected_box(self) -> None:
        if self.selected_box:
            self._delete_box(self.selected_box)

    def _delete_box(self, box: BoundingBox) -> None:
        if box in self.bounding_boxes:
            self.bounding_boxes.remove(box)
            if self.selected_box == box:
                self.selected_box = None
            self._update_display()

    def _change_box_class(self, box: BoundingBox, class_id: int) -> None:
        box.class_id = class_id
        self._update_display()

    def _load_model_suggestions(self) -> None:
        if not self.current_image_path:
            return

        try:
            if not self.model:
                model_path = self.app_config.active_learning.model
                if os.path.exists(model_path):
                    self.model = YOLO(model_path)
                else:
                    messagebox.showerror("Error", f"Model not found: {model_path}")
                    return

            results = self.model.predict(
                self.current_image_path,
                conf=self.app_config.inference.confidence_threshold,
                agnostic_nms=self.app_config.inference.agnostic_nms,
                half=self.app_config.inference.half,
                verbose=False,
            )

            if results and results[0].boxes is not None:
                # Remove existing suggestions
                self.bounding_boxes = [
                    b for b in self.bounding_boxes if not getattr(b, "suggested", False)
                ]

                boxes = results[0].boxes
                for i in range(len(boxes.xyxy)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    cls = int(boxes.cls[i].cpu().numpy())
                    
                    # cls mapping
                    cls_mapping = {
                        0: 0,  # Red unknown -> Red unknown
                        1: 1,  # Red 1 -> Red 1
                        2: 2,  # Red 3 -> Red 3
                        3: 0,  # Red 4 -> Red unknown
                        4: 3,  # Red Sentry -> Red Sentry
                        5: 4,  # Blue unknown -> Blue unknown
                        6: 5,  # Blue 1 -> Blue 1
                        7: 6,  # Blue 3 -> Blue 3
                        8: 4,  # Blue 4 -> Blue unknown
                        9: 7,  # Blue Sentry -> Blue Sentry
                    }
                    
                    cls = cls_mapping[cls]

                    suggested_box = BoundingBox(
                        int(x1), int(y1), int(x2), int(y2), cls, suggested=True
                    )
                    # Manually set suggested attribute since it may not be in constructor
                    suggested_box.suggested = True
                    self.bounding_boxes.append(suggested_box)

                self._update_display()
                print(f"Loaded {len(boxes.xyxy)} model suggestions")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model suggestions: {e}")

    def _on_mouse_wheel(self, event) -> None:
        # Zoom with mouse wheel
        if event.delta > 0 or event.num == 4:  # Zoom in
            self._zoom_in(event)
        else:  # Zoom out
            self._zoom_out(event)

    def _on_canvas_configure(self, event) -> None:
        """Called when canvas size changes (window resize)"""
        # Only update if we have an image and the event is for the canvas widget
        if self.current_image and event.widget == self.canvas:
            # Use after_idle to avoid multiple rapid calls during window resize
            self.root.after_idle(self._update_display)

    def _zoom_in(self, event=None) -> None:
        self._zoom_at_point(1.2, event)

    def _zoom_out(self, event=None) -> None:
        self._zoom_at_point(1.0 / 1.2, event)

    def _zoom_at_point(self, factor: float, event=None) -> None:
        """Zoom in or out while keeping the point under the mouse cursor stationary"""
        old_zoom = self.zoom_level
        self.zoom_level = max(0.1, min(old_zoom * factor, 5.0))
        self.user_has_zoomed = True

        # If we have mouse event, zoom toward mouse position
        if event and self.current_image:
            # Get mouse position in canvas coordinates
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)

            # Convert to image coordinates (before zoom)
            old_scale = self.scale_factor
            img_x = canvas_x / old_scale if old_scale > 0 else 0
            img_y = canvas_y / old_scale if old_scale > 0 else 0

            # Update display (this recalculates scale_factor)
            self._update_display()

            # Convert image coordinates to new canvas coordinates (after zoom)
            new_canvas_x = img_x * self.scale_factor
            new_canvas_y = img_y * self.scale_factor

            # Calculate how much to scroll to keep the image point under the mouse
            # We want: new_canvas_x - scroll_x = event.x (mouse position in widget)
            scroll_x = new_canvas_x - event.x
            scroll_y = new_canvas_y - event.y

            # Get canvas dimensions
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Get scroll region
            bbox = self.canvas.bbox("all")
            if bbox:
                scroll_width = bbox[2] - bbox[0]
                scroll_height = bbox[3] - bbox[1]

                # Normalize scroll position (0.0 to 1.0)
                if scroll_width > canvas_width:
                    norm_x = scroll_x / (scroll_width - canvas_width)
                    self.canvas.xview_moveto(max(0, min(1, norm_x)))
                
                if scroll_height > canvas_height:
                    norm_y = scroll_y / (scroll_height - canvas_height)
                    self.canvas.yview_moveto(max(0, min(1, norm_y)))
        else:
            self._update_display()

    def _zoom_fit(self) -> None:
        self.zoom_level = 1.0
        self.user_has_zoomed = False  # Reset user zoom flag
        self._update_display()

    def _update_image_counter(self) -> None:
        total = len(self.image_files)
        current = self.current_index + 1 if self.image_files else 0
        self.image_counter_label.config(text=f"{current}/{total}")

    def _update_image_info(self) -> None:
        if self.current_image_path:
            filename = os.path.basename(self.current_image_path)
            self.image_name_label.config(text=filename)

            # Update entropy display in toolbar
            if (
                self.active_learning_mode
                and self.current_image_path in self.entropy_values
            ):
                entropy = self.entropy_values[self.current_image_path]
                self.current_entropy = entropy
                self.entropy_label.config(text=f"Entropy: {entropy:.4f}")
            else:
                self.current_entropy = None
                self.entropy_label.config(text="")
        else:
            self.image_name_label.config(text="")
            self.current_entropy = None
            self.entropy_label.config(text="")

    def _update_bbox_list(self) -> None:
        self.bbox_listbox.delete(0, tk.END)
        for i, bbox in enumerate(self.bounding_boxes):
            class_name = self.data_config.names.get(bbox.class_id, str(bbox.class_id))
            is_suggested = getattr(bbox, "suggested", False)
            status = " (suggested)" if is_suggested else ""
            self.bbox_listbox.insert(tk.END, f"{i + 1}: {class_name}{status}")

    def _open_directory(self) -> None:
        directory = filedialog.askdirectory(title="Select Images Directory")
        if directory:
            self.app_config.raw_images_path = directory
            self._load_images()
            if self.image_files:
                self._load_current_image()

    def _generate_dataset(self) -> None:
        try:
            generate_dataset()
            messagebox.showinfo("Success", "Dataset generated successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate dataset: {e}")

    def _train_model(self) -> None:
        try:
            train_model()
            messagebox.showinfo("Success", "Model training completed!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {e}")

    def _run_active_learning(self) -> None:
        try:
            compute_low_confidence_images()
            messagebox.showinfo(
                "Success",
                "Active learning analysis completed!\n\nTip: Enable 'Use Active Learning List' to view suggested images.",
            )

            # Only refresh if already in active learning mode
            if self.active_learning_mode:
                self._load_images()
                if self.image_files:
                    self.current_index = 0
                    self._load_current_image()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run active learning: {e}")

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    app = LabelingTool()
    app.run()
