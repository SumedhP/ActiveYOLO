from collections import Counter
from typing import List, Tuple
from label import Label

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
        cls: int(count * val_ratio)
        for cls, count in total_class_counts.items()
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

    idxs = list(range(len(labels)))
    idxs.sort(key=_rarity_score, reverse=True)
    
    # Time for greed
    for idx in idxs:
        class_counts = image_class_counts[idx]
        
        fills_need = False
        for cls in class_counts:
            if current_val_counts[cls] < desired_class_counts[cls]:
                fills_need = True
                break
        
        if fills_need:
            val_set.append(labels[idx])
            current_val_counts.update(class_counts)
        else:
            train_set.append(labels[idx])
    
    return train_set, val_set
