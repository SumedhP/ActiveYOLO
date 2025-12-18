from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Annotation:
    id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class Label:
    file_path: str
    annotations: List[Annotation]

    @staticmethod
    def parse_label_file(file_path: str) -> "Label":
        annotations = []
        with open(file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split()

                if len(parts) != 5:
                    raise ValueError(f"Invalid label: {file_path}")

                id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                annotations.append(Annotation(id, x_center, y_center, width, height))
        return Label(file_path, annotations)

    def get_class_ids(self) -> Dict[int, int]:
        class_ids = {}
        for annotation in self.annotations:
            if annotation.id not in class_ids:
                class_ids[annotation.id] = 0
            class_ids[annotation.id] += 1
        return class_ids
