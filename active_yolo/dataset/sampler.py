from collections import Counter, defaultdict
import math
from typing import Dict, List, Tuple
from label import Label
import random

BACKGROUND_ID = -1


def stratified_split(
    labels: List[Label], val_ratio: float
) -> Tuple[List[Label], List[Label]]:
    # 1. Total class counts
    total_class_counts: Dict[int, int] = defaultdict(int)
    for label in labels:
        class_ids = label.get_class_ids()
        if not class_ids:
            total_class_counts[BACKGROUND_ID] += 1
        else:
            for class_id, count in label.get_class_ids().items():
                total_class_counts[class_id] += count

    # 2. Target validation counts per class
    target_val_counts = {
        class_id: math.ceil(count * val_ratio)
        for class_id, count in total_class_counts.items()
    }

    # 3. Current validation counts
    current_val_counts: Dict[int, int] = defaultdict(int)

    # 4. Sort labels by rarity & size
    def label_priority(label: Label) -> float:
        class_ids = label.get_class_ids()
        if not class_ids:
            return float("inf")  # Background image lowest priority

        score = 0.0
        for class_id, count in class_ids.items():
            # rarer classes get higher priority
            score += count / (total_class_counts[class_id] + 1e-6)
        return -score  # negative for descending sort

    sorted_labels = sorted(labels, key=label_priority)

    train_labels: List[Label] = []
    val_labels: List[Label] = []

    # 5. Greedy assignment
    for label in sorted_labels:
        class_ids = label.get_class_ids()

        if not class_ids:
            label_class_counts = {BACKGROUND_ID: 1}
        else:
            label_class_counts = class_ids

        can_go_to_val = True
        for class_id, count in label_class_counts.items():
            if current_val_counts[class_id] + count > target_val_counts[class_id]:
                can_go_to_val = False
                break

        if can_go_to_val:
            val_labels.append(label)
            for class_id, count in label_class_counts.items():
                current_val_counts[class_id] += count
        else:
            train_labels.append(label)

    print("Final validation class distribution:")
    for class_id, count in sorted(current_val_counts.items()):
        print(f"{class_id}: {count}")

    return train_labels, val_labels


def random_split(
    labels: List[Label], val_ratio: float
) -> Tuple[List[Label], List[Label]]:
    shuffled_labels = labels[:]
    random.shuffle(shuffled_labels)

    val_size = int(len(labels) * val_ratio)
    val_set = shuffled_labels[:val_size]
    train_set = shuffled_labels[val_size:]

    return train_set, val_set
