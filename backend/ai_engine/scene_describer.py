import cv2
import re
import time
import logging
import numpy as np
from backend.config import FORCE_MOCK_AI

logger = logging.getLogger(__name__)

# Try to import torch and transformers
try:
    import torch
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("torch or transformers (BLIP-2) not installed. Using simulated scene description.")

class SceneDescriber:
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = None
        self.is_mock = FORCE_MOCK_AI or not TRANSFORMERS_AVAILABLE
        self.last_description = "A room containing standard furniture and items."
        self.last_gray_frame = None
        
    def load_model(self):
        if self.is_mock:
            logger.info("Initializing simulated Scene Describer.")
            return True
            
        try:
            logger.info("Loading BLIP-2 model (this may take a few minutes)...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Using standard model loader (this can download ~10GB. If on CPU, we use float32, otherwise float16)
            self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b", 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.model.to(self.device)
            logger.info("BLIP-2 model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load BLIP-2: {e}. Switching to mock scene descriptions.")
            self.is_mock = True
            return True

    def detect_significant_change(self, frame, threshold=0.15):
        """
        Check if the scene has changed significantly compared to the last frame.
        Computes absolute pixel difference between grayscale frames.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (64, 64))
        
        if self.last_gray_frame is None:
            self.last_gray_frame = gray_resized
            return True
            
        # Absolute difference
        diff = cv2.absdiff(self.last_gray_frame, gray_resized)
        mean_diff = np.mean(diff) / 255.0
        
        self.last_gray_frame = gray_resized
        
        # If difference exceeds threshold, scene has changed significantly
        if mean_diff > threshold:
            logger.info(f"Significant scene change detected: diff {mean_diff:.3f} > {threshold}")
            return True
        return False

    def describe(self, frame):
        """
        Generates a natural language description of the frame.
        """
        if self.is_mock:
            return self._run_mock_description(frame)
            
        try:
            from PIL import Image
            raw_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            prompt = "Question: Describe this scene for a blind person in one clear sentence. Focus on: what is directly ahead, any obstacles, and what type of environment this is. Answer:"
            
            inputs = self.processor(raw_image, text=prompt, return_tensors="pt").to(self.device, torch.float16 if self.device == "cuda" else torch.float32)
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=40)
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            # Post-processing
            processed_desc = self.post_process(generated_text)
            self.last_description = processed_desc
            return processed_desc
        except Exception as e:
            logger.error(f"Error running BLIP-2 inference: {e}")
            return self._run_mock_description(frame)

    def post_process(self, text):
        """
        Post-process text outputs. Removes artifacts, limits length, adds spatial context.
        """
        # Strip out typical repetitive prefixes
        text = text.replace("Answer:", "").strip()
        
        # Lowercase first letter if it starts a sentence, clean spacing
        text = re.sub(r'^\w', lambda m: m.group(0).lower(), text)
        
        # Basic environment classifier based on keywords
        environment = "indoor"
        outdoor_keywords = ["street", "road", "sky", "grass", "tree", "building", "outside", "sidewalk", "park"]
        if any(keyword in text.lower() for keyword in outdoor_keywords):
            environment = "outdoor"
            
        # Truncate to max 25 words
        words = text.split()
        if len(words) > 25:
            text = " ".join(words[:25]) + "."
            
        prefix = f"You are in what appears to be an {environment} environment."
        return f"{prefix} {text}"

    def _run_mock_description(self, frame):
        """
        Generates simulated scene descriptions based on detections.
        """
        # Determine if night mode or dark
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        environment = "indoor"
        if mean_brightness < 40:
            env_desc = "a dimly lit room with some furniture silhouettes visible."
        else:
            env_desc = "a room containing an exit door, a table on the left, and a person standing directly ahead."
            
        prefix = f"You are in what appears to be an {environment} environment."
        return f"{prefix} Directly ahead is {env_desc}"
