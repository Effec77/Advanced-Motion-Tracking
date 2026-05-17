import numpy as np

def bbox_overlaps(boxes1, boxes2):
    """
    Calculate IoU (Intersection over Union) between two sets of bounding boxes.
    
    Args:
        boxes1: numpy array of shape (N, 4) in format [x1, y1, x2, y2]
        boxes2: numpy array of shape (M, 4) in format [x1, y1, x2, y2]
    
    Returns:
        numpy array of shape (N, M) containing IoU values
    """
    boxes1 = np.asarray(boxes1)
    boxes2 = np.asarray(boxes2)
    
    # Calculate areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Calculate intersection
    x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0])
    y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1])
    x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2])
    y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Calculate union
    union = area1[:, None] + area2 - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-10)
    
    return iou