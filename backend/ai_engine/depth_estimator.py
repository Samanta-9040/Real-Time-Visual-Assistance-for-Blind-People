import cv2
import logging
import numpy as np
from backend.config import FORCE_MOCK_AI, FRAME_WIDTH, FRAME_HEIGHT

logger = logging.getLogger(__name__)

# Try importing torch/torchvision
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torch not installed. Depth estimation will run in simulation mode.")

class DepthEstimator:
    def __init__(self):
        self.model = None
        self.transform = None
        self.device = None
        self.is_mock = FORCE_MOCK_AI or not TORCH_AVAILABLE
        
    def load_model(self):
        if self.is_mock:
            logger.info("Initializing simulated Depth Estimator.")
            return True
            
        try:
            logger.info("Loading MiDaS depth estimator model...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Using DPT_Hybrid or MiDaS_small (MiDaS_small is faster for CPUs)
            model_type = "MiDaS_small" 
            self.model = torch.hub.load("intel-isl/MiDaS", model_type)
            self.model.to(self.device)
            self.model.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform
                
            logger.info("MiDaS model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load MiDaS depth model: {e}. Switching to mock depth estimator.")
            self.is_mock = True
            return True

    def estimate(self, frame):
        """
        Estimate depth map from frame. Returns a normalized depth map (0-1 float float32).
        Large values represent closer distances.
        """
        if self.is_mock:
            return self._run_mock_depth(frame)
            
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = self.transform(img).to(self.device)
            
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                
            depth_map = prediction.cpu().numpy()
            
            # Normalize to 0.0 - 1.0
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            if depth_max - depth_min > 0:
                depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
            else:
                depth_norm = np.zeros_like(depth_map)
                
            return depth_norm
        except Exception as e:
            logger.error(f"Error running depth estimation: {e}")
            return self._run_mock_depth(frame)

    def get_center_depth_class(self, depth_map):
        """
        Classifies center region depth value:
        - NEAR: < 1m (normalized value close to 1.0, e.g. > 0.7)
        - MID: 1m - 3m (normalized value e.g. 0.3 - 0.7)
        - FAR: > 3m (normalized value close to 0.0, e.g. < 0.3)
        """
        h, w = depth_map.shape
        # Sample a center patch (10% of frame width/height)
        cy, cx = h // 2, w // 2
        dy, dx = h // 10, w // 10
        center_patch = depth_map[cy-dy:cy+dy, cx-dx:cx+dx]
        
        avg_depth = np.mean(center_patch)
        
        # High value = close in MiDaS
        if avg_depth > 0.7:
            return "NEAR", float(avg_depth)
        elif avg_depth > 0.35:
            return "MID", float(avg_depth)
        else:
            return "FAR", float(avg_depth)

    def cross_validate(self, depth_map, detections):
        """
        Cross-validate YOLO distance estimates with depth map data.
        Updates detections with validated distances.
        """
        h, w = depth_map.shape
        validated = []
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # Clip bounds
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h - 1, y2))
            
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                validated.append(det)
                continue
                
            # Sample bounding box region on depth map
            region_patch = depth_map[y1:y2, x1:x2]
            avg_depth = np.mean(region_patch) # 0 to 1
            
            # Convert depth map normalized score to distance (rough mapping: distance = k / avg_depth)
            # Let's say max depth 1.0 -> 0.5m, min depth 0.0 -> 10m
            # Map k: dist = 0.5 + (1.0 - avg_depth) * 5.0
            depth_dist = round(0.5 + (1.0 - avg_depth) * 5.0, 2)
            
            # Weighted average between YOLO geometric estimate and depth map estimate
            yolo_dist = det["distance_m"]
            
            # If YOLO estimate is close to depth estimate or YOLO is INFO priority, blend them
            # Else, trust depth map for close hazards
            blended_dist = round(0.4 * yolo_dist + 0.6 * depth_dist, 2)
            
            det["distance_m"] = blended_dist
            validated.append(det)
            
        return validated

    def _run_mock_depth(self, frame):
        """
        Simulate a depth map:
        - Creates a gradient representing vertical ground plane (closer at bottom, farther at top).
        - Overlays concentric circles representing objects (closer in center).
        """
        h, w = frame.shape[:2]
        
        # 1. Base perspective gradient (far at top, near at bottom)
        y_indices = np.arange(h).reshape(h, 1)
        base_gradient = (y_indices / h) * 0.7
        depth_map = np.repeat(base_gradient, w, axis=1)
        
        # 2. Add door (middle, MID range ~0.5 normalized)
        # Door coordinates in mock detector: [270, 150, 370, 350]
        depth_map[150:350, 270:370] = 0.52
        
        # 3. Add table (left bottom, NEAR range ~0.8 normalized)
        # Table coordinates: [50, 320, 200, 440]
        depth_map[320:440, 50:200] = 0.78
        
        # 4. Add person (right center, variable range depending on movement simulation)
        # We can look for colors or simply use current timestamp to place a simulated moving blob
        import time
        t = time.time()
        offset = (int(t * 15) % 120)
        p_x1 = int(450 - offset // 2)
        p_y1 = int(230 - offset // 3)
        p_x2 = int(550 + offset // 2)
        p_y2 = int(450 + offset // 3)
        
        # Clamp coordinates
        p_x1 = max(0, min(w-1, p_x1))
        p_x2 = max(0, min(w-1, p_x2))
        p_y1 = max(0, min(h-1, p_y1))
        p_y2 = max(0, min(h-1, p_y2))
        
        # Depth mapping based on box size
        # Larger box -> closer -> larger value
        person_depth = min(0.9, 0.4 + (offset / 120.0) * 0.5)
        depth_map[p_y1:p_y2, p_x1:p_x2] = person_depth
        
        return depth_map.astype(np.float32)
