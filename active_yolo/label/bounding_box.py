from label import Annotation


class BoundingBox:
    def __init__(self, x1: int, y1: int, x2: int, y2: int, class_id: int, suggested: bool = False):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.class_id = class_id
        self.suggested = suggested
        self.selected = False

    def contains_point(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def to_yolo_annotation(self, img_width: int, img_height: int) -> Annotation:
        x_center = ((self.x1 + self.x2) / 2) / img_width
        y_center = ((self.y1 + self.y2) / 2) / img_height
        width = (self.x2 - self.x1) / img_width
        height = (self.y2 - self.y1) / img_height
        return Annotation(self.class_id, x_center, y_center, width, height)

    @classmethod
    def from_yolo_annotation(cls, annotation: Annotation, img_width: int, img_height: int) -> "BoundingBox":
        x_center = annotation.x_center * img_width
        y_center = annotation.y_center * img_height
        width = annotation.width * img_width
        height = annotation.height * img_height
        
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        return cls(x1, y1, x2, y2, annotation.id)
