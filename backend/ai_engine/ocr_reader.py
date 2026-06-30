import cv2
import re
import time
import numpy as np
import logging
from backend.config import (
    OCR_CONFIDENCE_THRESHOLD, OCR_DEDUP_SIMILARITY, FORCE_MOCK_AI
)
from backend.utils.distance_calc import calculate_region
from backend.utils.dedup_utils import is_duplicate_text

logger = logging.getLogger(__name__)

# Try importing EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("easyocr not installed. Checking for pytesseract.")

# Try importing pytesseract
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    logger.warning("pytesseract not installed. OCR will run in simulation mode.")

class OCRReader:
    def __init__(self, languages=['en']):
        self.languages = languages
        self.reader = None
        self.is_mock = FORCE_MOCK_AI or (not EASYOCR_AVAILABLE and not PYTESSERACT_AVAILABLE)
        self.previous_texts = [] # List to track recently spoken text
        
    def load_model(self):
        if self.is_mock:
            logger.info("Initializing simulated OCR Reader.")
            return True
            
        if EASYOCR_AVAILABLE:
            try:
                logger.info(f"Loading EasyOCR reader for languages {self.languages}...")
                self.reader = easyocr.Reader(self.languages, gpu=False) # run on CPU by default
                logger.info("EasyOCR loaded successfully.")
                return True
            except Exception as e:
                logger.error(f"Failed to load EasyOCR: {e}. Checking pytesseract fallback.")
                
        if PYTESSERACT_AVAILABLE:
            try:
                # Test pytesseract installation
                pytesseract.get_tesseract_version()
                logger.info("pytesseract detected and verified.")
                return True
            except Exception as e:
                logger.error(f"pytesseract verification failed: {e}. Switching to mock OCR.")
                self.is_mock = True
                
        self.is_mock = True
        return True

    def preprocess_for_ocr(self, frame):
        """
        Preprocessing pipeline for OCR:
        1. Grayscale
        2. Adaptive thresholding
        3. Morphological closing to connect text regions
        4. Deskew (straighten text)
        """
        if frame is None:
            return None, None
            
        # 1. Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 3. Morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 4. Deskew
        # Find all threshold pixels
        coords = np.column_stack(np.where(closed > 0))
        angle = 0.0
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            # OpenCV angle representation check
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
        # Rotate image if angle is significant but not too large
        rotated_frame = frame
        if 0.5 < abs(angle) < 20.0:
            h, w = frame.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
        return rotated_frame, cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY)

    def classify_text(self, text):
        """
        Classifies OCR text into signage, price, number/address, or general text.
        """
        text_upper = text.upper()
        
        # Signage words
        signage_words = ["EXIT", "ENTER", "PUSH", "PULL", "OPEN", "CLOSED", "WAY", "RESTROOM", "TOILET", "LADIES", "GENTS"]
        if any(word in text_upper for word in signage_words):
            return "signage"
            
        # Price pattern ($X.XX, Rs X.XX, etc.)
        price_pattern = r"(\$|rs|usd|£|€)?\s?\d+[\.,]\d{2}"
        if re.search(price_pattern, text_upper) or any(p in text_upper for p in ["PRICE", "COST", "SALE"]):
            return "price"
            
        # Number / Address pattern
        number_pattern = r"\b\d{2,5}\b"
        if re.search(number_pattern, text_upper) or "STREET" in text_upper or "AVE" in text_upper or "ROAD" in text_upper:
            return "number/address"
            
        return "general text"

    def read_text(self, frame):
        """
        Executes OCR on the frame.
        """
        if self.is_mock:
            return self._run_mock_ocr()
            
        # Preprocess frame
        rotated_frame, gray_frame = self.preprocess_for_ocr(frame)
        results = []
        
        if EASYOCR_AVAILABLE and self.reader:
            try:
                # Run EasyOCR
                ocr_out = self.reader.readtext(rotated_frame)
                for bbox_coords, text, conf in ocr_out:
                    if conf < OCR_CONFIDENCE_THRESHOLD or len(text.strip()) <= 2:
                        continue
                        
                    # EasyOCR bbox format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    # Convert to [x1, y1, x2, y2]
                    xs = [pt[0] for pt in bbox_coords]
                    ys = [pt[1] for pt in bbox_coords]
                    bbox = [min(xs), min(ys), max(xs), max(ys)]
                    
                    if is_duplicate_text(text, self.previous_texts, OCR_DEDUP_SIMILARITY):
                        continue
                        
                    category = self.classify_text(text)
                    region = calculate_region(bbox)
                    
                    results.append({
                        "text": text.strip(),
                        "confidence": float(conf),
                        "bbox": bbox,
                        "category": category,
                        "region": region
                    })
            except Exception as e:
                logger.error(f"EasyOCR reader execution error: {e}. Trying pytesseract fallback.")
                
        # Fallback to pytesseract if EasyOCR failed or isn't loaded
        if len(results) == 0 and PYTESSERACT_AVAILABLE:
            try:
                # Fetch text bounding box details from pytesseract
                # image_to_data returns dataframe-like string dict
                data = pytesseract.image_to_data(gray_frame, output_type=pytesseract.Output.DICT)
                n_boxes = len(data['level'])
                for i in range(n_boxes):
                    text = data['text'][i].strip()
                    conf = float(data['conf'][i]) / 100.0 # pytesseract confidence is 0-100
                    
                    if conf < OCR_CONFIDENCE_THRESHOLD or len(text) <= 2:
                        continue
                        
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    bbox = [x, y, x + w, y + h]
                    
                    if is_duplicate_text(text, self.previous_texts, OCR_DEDUP_SIMILARITY):
                        continue
                        
                    category = self.classify_text(text)
                    region = calculate_region(bbox)
                    
                    results.append({
                        "text": text,
                        "confidence": conf,
                        "bbox": bbox,
                        "category": category,
                        "region": region
                    })
            except Exception as e:
                logger.error(f"pytesseract OCR execution error: {e}")
                
        # Maintain a queue of last 10 OCR items to prevent duplication
        for res in results:
            self.previous_texts.append(res["text"])
        if len(self.previous_texts) > 20:
            self.previous_texts = self.previous_texts[-20:]
            
        return results

    def _run_mock_ocr(self):
        """
        Generate mock text detection.
        Matches coordinates of the mock yellow exit sign drawn in CameraManager.
        """
        t = time.time()
        # Simulate text detection on the sign drawn on mock canvas
        # Yellow exit sign coordinates: [295, 120] -> exit text size ~ 100x30
        results = [{
            "text": "EXIT",
            "confidence": 0.98,
            "bbox": [290, 95, 360, 125],
            "category": "signage",
            "region": "CENTER"
        }]
        
        # Add a simulated price tag every few seconds to show different tags
        if int(t) % 15 < 5:
            results.append({
                "text": "$4.99",
                "confidence": 0.89,
                "bbox": [110, 390, 170, 415],
                "category": "price",
                "region": "LEFT"
            })
            
        return results
