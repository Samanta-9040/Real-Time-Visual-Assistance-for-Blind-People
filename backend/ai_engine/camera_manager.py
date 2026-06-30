import cv2
import time
import numpy as np
import logging
from datetime import datetime
from backend.config import FRAME_WIDTH, FRAME_HEIGHT, BLUR_THRESHOLD

logger = logging.getLogger(__name__)

class CameraManager:
    def __init__(self, source_type="webcam", source_path=0, fps=10, night_mode=False):
        self.source_type = source_type
        self.source_path = source_path
        self.fps = fps
        self.night_mode = night_mode
        self.cap = None
        self.is_running = False
        self.frame_delay = 1.0 / self.fps
        
    def start(self):
        self.is_running = True
        if self.source_type == "mock":
            logger.info("Initializing camera manager in MOCK mode.")
            return True
            
        logger.info(f"Starting camera source: {self.source_type} ({self.source_path})")
        try:
            if self.source_type == "webcam":
                # Convert source_path to int if it looks like an integer index
                idx = int(self.source_path) if str(self.source_path).isdigit() else 0
                self.cap = cv2.VideoCapture(idx)
            elif self.source_type in ["ip_camera", "upload"]:
                self.cap = cv2.VideoCapture(self.source_path)
            
            if self.cap and self.cap.isOpened():
                # Set capture resolution if possible
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH if hasattr(cv2, 'CAP_PROP_FRAME_WIDTH') else 3, FRAME_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT if hasattr(cv2, 'CAP_PROP_FRAME_HEIGHT') else 4, FRAME_HEIGHT)
                return True
            else:
                logger.error("Failed to open camera source. Falling back to MOCK mode.")
                self.source_type = "mock"
                return True
        except Exception as e:
            logger.error(f"Error starting camera: {e}. Falling back to MOCK mode.")
            self.source_type = "mock"
            return True

    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("Camera manager stopped.")

    def preprocess_frame(self, frame):
        """
        Preprocessing Pipeline:
        1. Resize to 640x480
        2. CLAHE (Contrast Limited Adaptive Histogram Equalization) for low-light
        3. Gaussian denoise (kernel 3x3)
        4. Auto white balance correction
        5. Night mode adjustments if enabled
        """
        if frame is None:
            return None
            
        # 1. Resize
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        
        # 5. Night Mode Toggle (prior to histogram equalization to scale gains)
        if self.night_mode:
            # Simple digital gain boost for mock night vision
            frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)
            
        # 2. CLAHE on Lab color space to preserve color details
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # 3. Gaussian denoise
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        
        # 4. Simple Auto White Balance (Gray World assumption)
        result = frame.copy()
        cols, rows, ch = result.shape
        avg_b = np.average(result[:, :, 0])
        avg_g = np.average(result[:, :, 1])
        avg_r = np.average(result[:, :, 2])
        avg_gray = (avg_b + avg_g + avg_r) / 3.0
        
        if avg_b > 0 and avg_g > 0 and avg_r > 0:
            kb = avg_gray / avg_b
            kg = avg_gray / avg_g
            kr = avg_gray / avg_r
            result[:, :, 0] = np.clip(result[:, :, 0] * kb, 0, 255).astype(np.uint8)
            result[:, :, 1] = np.clip(result[:, :, 1] * kg, 0, 255).astype(np.uint8)
            result[:, :, 2] = np.clip(result[:, :, 2] * kr, 0, 255).astype(np.uint8)
            frame = result

        return frame

    def calculate_metadata(self, frame):
        """
        Calculate frame brightness and blur score (Laplacian variance).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Brightness: average gray pixel value
        brightness = float(np.mean(gray))
        
        # Blur score: Laplacian variance
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "brightness_level": brightness,
            "blur_score": blur_score
        }

    def generate_mock_frame(self):
        """
        Generate a synthetic environment frame for mock source mode.
        """
        # Create a dark/neon background representing indoor workspace
        img = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        # Background gradient
        for y in range(FRAME_HEIGHT):
            img[y, :, 0] = int(20 + 20 * (y / FRAME_HEIGHT)) # Blue channel
            img[y, :, 1] = int(10 + 10 * (y / FRAME_HEIGHT)) # Green
            img[y, :, 2] = int(15 + 15 * (y / FRAME_HEIGHT)) # Red
            
        # Draw some decorative abstract elements (like outlines of furniture/walls)
        cv2.line(img, (50, 400), (250, 250), (40, 40, 40), 2)
        cv2.line(img, (250, 250), (450, 250), (40, 40, 40), 2)
        cv2.line(img, (450, 250), (600, 400), (40, 40, 40), 2)
        
        # Let's draw a mock "door" outline in the middle
        cv2.rectangle(img, (270, 150), (370, 350), (80, 80, 80), 2)
        cv2.circle(img, (355, 250), 4, (120, 120, 120), -1) # doorknob
        
        # Draw a table outline
        cv2.rectangle(img, (50, 320), (200, 440), (60, 60, 60), 2)
        cv2.line(img, (50, 440), (50, 480), (60, 60, 60), 2)
        cv2.line(img, (200, 440), (200, 480), (60, 60, 60), 2)
        
        # Put some text on the wall
        cv2.putText(img, "EXIT", (295, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Place a mock "person" silhouette or circle
        cv2.circle(img, (500, 200), 30, (50, 50, 150), -1)
        cv2.rectangle(img, (450, 230), (550, 450), (50, 50, 150), -1)
        
        return img

    def get_frames(self):
        """
        Generator yielding (preprocessed_frame, metadata).
        Enforces configured FPS.
        """
        while self.is_running:
            start_time = time.time()
            frame = None
            
            if self.source_type == "mock":
                frame = self.generate_mock_frame()
            elif self.cap:
                ret, raw_frame = self.cap.read()
                if ret:
                    frame = raw_frame
                else:
                    # End of file or disconnected webcam
                    logger.warning("Camera stream read error or end of video file. Restarting source/mocking.")
                    if self.source_type == "upload":
                        # Restart video file
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, raw_frame = self.cap.read()
                        if ret:
                            frame = raw_frame
                    else:
                        frame = self.generate_mock_frame()
            
            if frame is not None:
                metadata = self.calculate_metadata(frame)
                
                # Check blur score
                if metadata["blur_score"] < BLUR_THRESHOLD and self.source_type != "mock":
                    logger.warning(f"Blurry frame skipped: blur_score {metadata['blur_score']:.2f} < {BLUR_THRESHOLD}")
                    # Still yield frame but maybe mark it blurry or skip. 
                    # Prompt says: "If blur_score < threshold, skip frame and log 'blurry frame skipped'"
                    time.sleep(max(0, self.frame_delay - (time.time() - start_time)))
                    continue
                
                preprocessed = self.preprocess_frame(frame)
                yield preprocessed, metadata
            else:
                time.sleep(0.01)
                
            # Control frame rate
            elapsed = time.time() - start_time
            sleep_time = self.frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
