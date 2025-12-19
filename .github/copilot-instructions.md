# ActiveYOLO Copilot Instructions

## Project Overview
ActiveYOLO is an active learning pipeline for YOLOv11 model training using Ultralytics. The system iteratively improves by training models, analyzing uncertainty, and suggesting the most valuable images for manual annotation.

## Architecture Components

### Core Pipeline Flow
1. **Raw Images** → **Manual Annotation** → **Dataset Generation** → **Model Training** → **Active Learning Analysis** → **Repeat**
2. Three YAML configs control behavior: `configs/app.yaml` (paths, active learning params), `configs/data.yaml` (class mappings), `configs/train.yaml` (YOLO + SSL training params)
3. Entry points: [active_yolo/main.py](active_yolo/main.py) loads all configs for validation; actual work done via individual scripts

### Key Modules
- **[active_yolo/generate_dataset.py](active_yolo/generate_dataset.py)**: Converts labels → balanced YOLO train/val splits using stratified sampling
- **[active_yolo/train.py](active_yolo/train.py)**: Handles YOLO training + optional Lightly SSL backbone pretraining 
- **[active_yolo/active_learning.py](active_yolo/active_learning.py)**: Analyzes model uncertainty via entropy + K-means clustering to suggest next images
- **[active_yolo/label/](active_yolo/label/)**: YOLO format label parsing (`class_id x_center y_center width height`)

### Active Learning Strategy
Uses multi-step uncertainty analysis in [active_yolo/active_learning/](active_yolo/active_learning/):
1. **Entropy calculation**: Bernoulli entropy from detection confidences (`p * -log(p) + (1-p) * -log(1-p)`)
2. **Embedding extraction**: YOLO model embeddings via `.embed()` method 
3. **K-means clustering**: Groups similar images, selects highest entropy from each cluster
4. **Output**: [output/low_confidence_images.txt](output/low_confidence_images.txt) with sorted image paths + entropy scores

### Configuration Pattern
All configs use dataclass wrappers with `load_from_yaml()` static methods:
- [active_yolo/config/app_config.py](active_yolo/config/app_config.py): Nested dataclasses for `DatasetConfig`, `ActiveLearningConfig`, `InferenceConfig`
- Configs loaded via `Config.load_config()` which hardcodes `configs/*.yaml` paths

### Dataset Generation Logic
[active_yolo/dataset/sampler.py](active_yolo/dataset/sampler.py) implements stratified splitting:
- Calculates class rarity scores (`1.0 / total_class_counts[cls]`)
- Greedy allocation prioritizing rare classes for validation set
- Preserves class balance while respecting `val_split` ratio

## Development Patterns

### Multiprocessing Convention
Performance-critical operations use `tqdm.contrib.concurrent.process_map`:
```python
from functools import partial
from tqdm.contrib.concurrent import process_map
worker_func = partial(_worker_function, model_path)
results = process_map(worker_func, chunks, max_workers=num_processes)
```

### YOLO Integration
- Models auto-export to TensorRT via `model.export(format="engine", nms=True)` for inference speedup
- Use `conf=1e-3` for low-confidence detections in active learning
- TensorBoard logging enabled via `settings.update({"tensorboard": True})`

### Error Handling
Limited error handling - focuses on continuation over robustness (see TensorRT export fallback pattern)

## Key Commands & Workflows

### Training Workflow
```bash
# Generate balanced dataset from labels
python -m active_yolo.generate_dataset

# Train YOLO model 
python -m active_yolo.train

# Optional: Self-supervised backbone pretraining
python -m active_yolo.train --backbone

# Analyze uncertainty for next annotation batch
python -m active_yolo.active_learning
```

### File Structure Expectations
- Raw images: `raw_images/*.jpg` (configurable)
- Labels: `labels/*.txt` (YOLO format, one per image)
- Generated dataset: `dataset/` (created by generate_dataset.py)
- Model output: `models/best.pt` (default active learning model)

## Dependencies & Environment
- **Core**: `ultralytics>=8.3.240`, `lightly-train[ultralytics]>=0.13.0`
- **Performance**: `tensorrt>=10.14.1.48.post1`, `onnxruntime-gpu>=1.23.2`
- **ML**: `scikit-learn>=1.8.0` (K-means clustering)
- Python 3.13+ required per [pyproject.toml](pyproject.toml)

## Planned Features
Per [docs/DESIGN DOC.md](docs/DESIGN%20DOC.md):
- Tkinter GUI for annotation with active learning mode
- Model-suggested bounding box pre-population 
- Menu integration for dataset/training pipeline triggers
