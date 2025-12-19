import numpy as np
from ultralytics import YOLO # type: ignore[reportPrivateImportUsage]
from ultralytics.engine.results import Results


def compute_entropy(model: YOLO, image_path: str) -> float:
    results = model.predict(image_path, conf=1e-3, verbose=False)

    actual_result = None
    for result in results:
        if isinstance(result, Results):
            actual_result = result
            break

    if actual_result is None:
        print(f"No valid results for image: {image_path}")
        return 0.0

    entropy = 0.0
    if actual_result.boxes is None or len(actual_result.boxes) == 0:
        # Low-priority for images with no detections
        return entropy

    if actual_result.boxes.conf is None or len(actual_result.boxes.conf) == 0:
        return entropy

    confidences = actual_result.boxes.conf.numpy()  # type: ignore[possibly-missing-attribute]
    if len(confidences) == 0:
        return entropy

    eps = 1e-10
    p = np.clip(confidences, eps, 1 - eps)
    # Bernoulli mean entropy
    entropy = p * -np.log(p) + (1 - p) * -np.log(1 - p)
    entropy = float(np.mean(entropy))
    return entropy
