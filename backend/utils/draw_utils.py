import cv2
import numpy as np

# Color mappings in BGR
PRIORITY_COLORS = {
    "CRITICAL": (0, 0, 255),    # Red
    "WARNING": (0, 165, 255),  # Orange
    "INFO": (255, 0, 0)        # Blue
}

FACE_COLOR = (255, 0, 255)     # Purple
OCR_COLOR = (255, 255, 0)      # Cyan

def draw_detections(frame, detections):
    """
    Draw colored bounding boxes with labels, distances, and regions on the frame.
    """
    annotated = frame.copy()
    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue
        
        x1, y1, x2, y2 = [int(v) for v in bbox]
        label = det.get("label", "unknown")
        dist = det.get("distance_m", 0.0)
        region = det.get("region", "CENTER")
        priority = det.get("priority", "INFO")
        
        color = PRIORITY_COLORS.get(priority, PRIORITY_COLORS["INFO"])
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Label string
        label_str = f"{label.upper()} {dist}m ({region})"
        
        # Text size and background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label_str, font, font_scale, thickness)
        
        # Draw text background
        cv2.rectangle(annotated, (x1, y1 - text_h - 6), (x1 + text_w + 4, y1), color, -1)
        # Draw text
        cv2.putText(annotated, label_str, (x1 + 2, y1 - 4), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
    return annotated

def draw_ocr(frame, ocr_results):
    """
    Draw bounding boxes and recognized text for OCR detections.
    """
    annotated = frame.copy()
    for ocr in ocr_results:
        bbox = ocr.get("bbox", [])
        if len(bbox) != 4:
            continue
            
        x1, y1, x2, y2 = [int(v) for v in bbox]
        text = ocr.get("text", "")
        category = ocr.get("category", "general text")
        
        # Limit text length shown on bounding box
        display_text = f"[{category}] {text[:15]}..." if len(text) > 15 else f"[{category}] {text}"
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), OCR_COLOR, 1)
        
        # Text size and background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)
        
        # Draw text background
        cv2.rectangle(annotated, (x1, y1 - text_h - 4), (x1 + text_w + 2, y1), OCR_COLOR, -1)
        # Draw text
        cv2.putText(annotated, display_text, (x1 + 1, y1 - 2), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        
    return annotated

def draw_faces(frame, faces):
    """
    Draw purple bounding boxes around faces and write recognized name.
    """
    annotated = frame.copy()
    for face in faces:
        bbox = face.get("bbox", [])
        if len(bbox) != 4:
            continue
            
        x1, y1, x2, y2 = [int(v) for v in bbox]
        name = face.get("name", "Unknown")
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), FACE_COLOR, 2)
        
        # Text size and background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        label_str = f"FACE: {name}"
        (text_w, text_h), baseline = cv2.getTextSize(label_str, font, font_scale, thickness)
        
        # Draw text background
        cv2.rectangle(annotated, (x1, y1 - text_h - 6), (x1 + text_w + 4, y1), FACE_COLOR, -1)
        # Draw text
        cv2.putText(annotated, label_str, (x1 + 2, y1 - 4), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
    return annotated

def generate_depth_heatmap(depth_map):
    """
    Convert a single channel depth map (0-1) to an RGB color heatmap
    where near is red and far is blue.
    """
    # Normalize map to 0-255
    depth_u8 = (depth_map * 255).astype(np.uint8)
    
    # Invert so near objects (large depth value in some models, or smaller in others) are red
    # Let's map high value -> Red, low value -> Blue
    # cv2.applyColorMap needs uint8
    heatmap = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
    return heatmap

def blend_depth_map(frame, depth_map, alpha=0.4):
    """
    Blend depth heatmap overlay on top of frame.
    """
    # Resize depth map to match frame size
    depth_resized = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
    
    # Generate heatmap
    heatmap = generate_depth_heatmap(depth_resized)
    
    # Blend images
    blended = cv2.addWeighted(frame, 1.0 - alpha, heatmap, alpha, 0)
    return blended
