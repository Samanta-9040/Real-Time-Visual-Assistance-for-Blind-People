from backend.config import KNOWN_HEIGHTS, FOCAL_LENGTH, FRAME_WIDTH

def calculate_region(bbox, frame_width=FRAME_WIDTH):
    """
    Given a bounding box [x1, y1, x2, y2], calculate if it is in
    the LEFT, CENTER, or RIGHT horizontal third of the frame.
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    
    third = frame_width / 3.0
    if center_x < third:
        return "LEFT"
    elif center_x < 2 * third:
        return "CENTER"
    else:
        return "RIGHT"

def estimate_distance(label, bbox):
    """
    Estimate the distance to the object in meters.
    Formula: distance = (known_height * focal_length) / pixel_height
    """
    x1, y1, x2, y2 = bbox
    pixel_height = max(1.0, y2 - y1) # Prevent division by zero
    
    known_height = KNOWN_HEIGHTS.get(label, 0.5) # Default height 0.5m if unknown
    
    distance = (known_height * FOCAL_LENGTH) / pixel_height
    return round(distance, 2)
