def iou(box_a, box_b):
    """
    Compute Intersection over Union (IoU) of two bounding boxes.
    Each box is in the format [x_min, y_min, x_max, y_max].
    """
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])
    intersection = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    if union == 0:
        return 0.0
    else:
        return intersection / union