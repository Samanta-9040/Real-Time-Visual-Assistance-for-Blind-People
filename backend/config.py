import os

# Base directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Database Configuration
DATABASE_PATH = os.path.join(BASE_DIR, "..", "database", "visionbridge.db")
SQLALCHEMY_DATABASE_URI = f"sqlite:///{DATABASE_PATH}"

# Directory Paths
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "..", "known_faces")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# Ensure directories exist
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Simulation/Fallback Settings
# Set to True to force simulated AI outputs even if models are installed
FORCE_MOCK_AI = False 

# Camera Settings
DEFAULT_SOURCE = "webcam" # "webcam", "ip_camera", "upload", or "mock"
DEFAULT_FPS = 10
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
BLUR_THRESHOLD = 15.0 # Laplacian variance threshold

# Object Detector Settings
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.5
FOCAL_LENGTH = 500.0 # Pixels

# Known average object heights in meters
KNOWN_HEIGHTS = {
    "person": 1.7,
    "car": 1.5,
    "bus": 3.0,
    "truck": 2.8,
    "bicycle": 1.0,
    "motorcycle": 1.0,
    "stairs": 2.0,
    "door": 2.0,
    "chair": 0.8,
    "table": 0.75,
    "dog": 0.5,
    "cat": 0.25,
    "pothole": 0.1,
    "fire hydrant": 0.9,
    "bottle": 0.25,
    "cup": 0.12,
    "laptop": 0.25,
    "phone": 0.15,
    "plant": 0.4
}

# OCR Settings
OCR_INTERVAL_FRAMES = 10
OCR_CONFIDENCE_THRESHOLD = 0.6
OCR_DEDUP_SIMILARITY = 0.85
OCR_LANGUAGES = ['en'] # Add 'hi' for Hindi, 'or' for Odia

# Scene Describer Settings
SCENE_INTERVAL_FRAMES = 30
SCENE_PIXEL_DIFF_THRESHOLD = 30.0 # Percentage change to re-run

# Depth Estimator Settings
DEPTH_INTERVAL_FRAMES = 5

# Audio Engine Settings
SPEECH_RATE = 160
SPEECH_VOLUME = 1.0
DEDUP_OBJECT_SECONDS = 4.0
DEDUP_TEXT_SECONDS = 10.0
DEFAULT_MODE = "standard" # "verbose", "standard", "minimal"
