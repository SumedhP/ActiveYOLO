from collections import Counter
from typing import List, Tuple
from label import Label
import random


def stratified_split(
    labels: List[Label], val_ratio: float
) -> Tuple[List[Label], List[Label]]:
    total_class_counts = Counter()
    image_class_counts = []

    for label in labels:
        class_counts = label.get_class_ids()
        total_class_counts.update(class_counts)
        image_class_counts.append(class_counts)

    desired_class_counts = {
        cls: int(count * val_ratio) for cls, count in total_class_counts.items()
    }

    current_val_counts = Counter()
    val_set = []
    train_set = []

    def _rarity_score(idx):
        # Calculate the rarity score based on the class counts
        # Something feels fishy about this logic
        score = 0.0
        for cls in image_class_counts[idx]:
            score += 1.0 / total_class_counts[cls]
        return score

    indices = list(range(len(labels)))
    indices.sort(key=_rarity_score, reverse=True)

    val_set_indices = []

    # Time for greed
    for idx in indices:
        class_counts = image_class_counts[idx]

        fills_need = False
        for cls in class_counts:
            if current_val_counts[cls] < desired_class_counts[cls]:
                fills_need = True
                break

        if fills_need:
            val_set_indices.append(idx)
            current_val_counts.update(class_counts)

    # If we don't have enough labels in the validation set, add more:
    while len(val_set_indices) < int(len(labels) * val_ratio):
        for idx in indices:
            if idx not in val_set_indices:
                val_set_indices.append(idx)
                break

    val_set = [labels[idx] for idx in val_set_indices]
    train_set = [label for label in labels if label not in val_set]

    return train_set, val_set


def random_split(
    labels: List[Label], val_ratio: float
) -> Tuple[List[Label], List[Label]]:
    shuffled_labels = labels[:]
    random.shuffle(shuffled_labels)

    val_size = int(len(labels) * val_ratio)
    val_set = shuffled_labels[:val_size]
    train_set = shuffled_labels[val_size:]

    return train_set, val_set
