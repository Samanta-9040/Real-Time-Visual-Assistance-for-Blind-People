import os
import time
import logging
import random
from backend.config import (
    YOLO_MODEL_PATH, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, 
    FORCE_MOCK_AI, FRAME_WIDTH, FRAME_HEIGHT
)
from backend.utils.distance_calc import calculate_region, estimate_distance

logger = logging.getLogger(__name__)

# Try to import ultralytics, set fallback flag if not installed
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("ultralytics (YOLO) not installed. Using simulated object detector.")

# Priority definitions
PRIORITY_GROUPS = {
    "CRITICAL": ["person", "car", "bus", "truck", "bicycle", "motorcycle", "stairs", "door"],
    "WARNING": ["chair", "table", "dog", "cat", "pothole", "fire hydrant"],
    "INFO": ["bottle", "cup", "laptop", "phone", "plant", "backpack", "umbrella", "handbag", "book"]
}

def get_priority(label):
    for priority, labels in PRIORITY_GROUPS.items():
        if label in labels:
            return priority
    return "INFO"

class ObjectDetector:
    def __init__(self):
        self.model = None
        self.is_mock = not ULTRALYTICS_AVAILABLE or FORCE_MOCK_AI
        self.tracker_id_counter = 0
        self.tracked_objects = {} # dict of id -> {bbox, label, last_seen, area}
        
    def load_model(self):
        if self.is_mock:
            logger.info("Initializing simulated Object Detector.")
            return True
            
        try:
            logger.info(f"Loading YOLOv8 model from {YOLO_MODEL_PATH}...")
            # Ultralytics will auto-download 'yolov8n.pt' to current folder if it doesn't exist
            # We enforce saving to models/yolov8n.pt
            self.model = YOLO(YOLO_MODEL_PATH)
            logger.info("YOLOv8 model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}. Switching to mock mode.")
            self.is_mock = True
            return True

    def _calculate_iou(self, boxA, boxB):
        # Determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # Compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # Compute the intersection over union
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def track_and_detect_movement(self, current_detections):
        """
        Simple centroid/IOU tracker to track objects and check if they are moving
        towards the camera (bounding box growing).
        """
        now = time.time()
        updated_detections = []
        new_tracked_objects = {}

        for det in current_detections:
            bbox = det["bbox"]
            label = det["label"]
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            
            # Find best match in tracked objects
            best_id = None
            best_iou = 0.3 # Minimum IoU overlap to match
            
            for tid, tobj in self.tracked_objects.items():
                if tobj["label"] == label:
                    iou = self._calculate_iou(bbox, tobj["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_id = tid
            
            is_moving = False
            is_approaching = False
            
            if best_id is not None:
                # Matched existing tracked object
                prev_obj = self.tracked_objects[best_id]
                prev_area = prev_obj["area"]
                
                # Check if bounding box area is growing (closer / approaching)
                # Significant growth over a frame represents approaching
                if area > 1.12 * prev_area:
                    is_moving = True
                    is_approaching = True
                elif area < 0.88 * prev_area:
                    is_moving = True
                
                tracking_id = best_id
            else:
                # Create a new tracked object
                self.tracker_id_counter += 1
                tracking_id = self.tracker_id_counter
            
            # Register in current frame tracking
            new_tracked_objects[tracking_id] = {
                "bbox": bbox,
                "label": label,
                "last_seen": now,
                "area": area
            }
            
            # Append tracking state to output
            det["id"] = tracking_id
            det["is_moving"] = is_moving
            det["is_approaching"] = is_approaching
            
            updated_detections.append(det)
            
        # Clean up old tracked objects not seen in last 2 seconds
        for tid, tobj in self.tracked_objects.items():
            if now - tobj["last_seen"] < 2.0 and tid not in new_tracked_objects:
                new_tracked_objects[tid] = tobj
                
        self.tracked_objects = new_tracked_objects
        return updated_detections

    def detect(self, frame):
        """
        Runs object detection on the frame.
        Returns a list of DetectionResult dicts.
        """
        detections = []
        
        if self.is_mock:
            detections = self._run_mock_detector()
        else:
            try:
                results = self.model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
                if results and len(results) > 0:
                    result = results[0]
                    boxes = result.boxes
                    for box in boxes:
                        # Extract bbox coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = self.model.names[cls_id]
                        
                        bbox = [x1, y1, x2, y2]
                        region = calculate_region(bbox)
                        distance = estimate_distance(label, bbox)
                        priority = get_priority(label)
                        
                        detections.append({
                            "label": label,
                            "confidence": conf,
                            "bbox": bbox,
                            "region": region,
                            "distance_m": distance,
                            "priority": priority,
                            "is_moving": False
                        })
            except Exception as e:
                logger.error(f"Error running YOLO detection: {e}. Switching to mock outputs.")
                detections = self._run_mock_detector()

        # Run tracking to add ids and detect movement (approaching)
        tracked_detections = self.track_and_detect_movement(detections)
        return tracked_detections

    def _run_mock_detector(self):
        """
        Generate simulated detections based on mock canvas parameters.
        Simulates some typical indoor objects moving.
        """
        detections = []
        t = time.time()
        
        # 1. Door (Static, Center-Left, mid range)
        detections.append({
            "label": "door",
            "confidence": 0.92,
            "bbox": [270, 150, 370, 350],
            "region": "CENTER",
            "distance_m": 2.8,
            "priority": "CRITICAL"
        })
        
        # 2. Table (Static, Left, near range)
        detections.append({
            "label": "table",
            "confidence": 0.88,
            "bbox": [50, 320, 200, 440],
            "region": "LEFT",
            "distance_m": 1.1,
            "priority": "WARNING"
        })
        
        # 3. Bottle on table (Static, Left, near range)
        detections.append({
            "label": "bottle",
            "confidence": 0.79,
            "bbox": [100, 280, 130, 320],
            "region": "LEFT",
            "distance_m": 1.0,
            "priority": "INFO"
        })
        
        # 4. Simulated Person (approaching slowly)
        # Bounding box expands over time using sine function
        offset = (int(t * 15) % 120)
        p_x1 = 450 - offset // 2
        p_y1 = 230 - offset // 3
        p_x2 = 550 + offset // 2
        p_y2 = 450 + offset // 3
        
        bbox_person = [p_x1, p_y1, p_x2, p_y2]
        region_person = calculate_region(bbox_person)
        distance_person = estimate_distance("person", bbox_person)
        
        detections.append({
            "label": "person",
            "confidence": 0.95,
            "bbox": bbox_person,
            "region": region_person,
            "distance_m": distance_person,
            "priority": "CRITICAL"
        })
        
        return detections
