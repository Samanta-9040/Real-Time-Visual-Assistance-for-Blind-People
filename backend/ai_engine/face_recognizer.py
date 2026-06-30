import os
import json
import logging
import numpy as np
from backend.config import KNOWN_FACES_DIR, FORCE_MOCK_AI

logger = logging.getLogger(__name__)

# Try to import face_recognition
try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    logger.warning("face_recognition or dlib not installed. Using simulated face recognizer.")

class FaceRecognizer:
    def __init__(self):
        self.is_mock = FORCE_MOCK_AI or not FACE_REC_AVAILABLE
        self.known_encodings = [] # List of numpy arrays
        self.known_names = []      # List of strings
        
    def load_known_faces(self, db_session=None):
        """
        Load all face encodings from the SQLite database.
        """
        self.known_encodings = []
        self.known_names = []
        
        if db_session is None:
            logger.warning("Database session not provided to face recognizer. Running in empty face registry.")
            return True
            
        try:
            from backend.models import KnownFace
            faces = db_session.query(KnownFace).all()
            for face in faces:
                try:
                    encoding = np.array(json.loads(face.encoding_json))
                    self.known_encodings.append(encoding)
                    self.known_names.append(face.name)
                except Exception as ex:
                    logger.error(f"Error decoding face record {face.name}: {ex}")
            logger.info(f"Loaded {len(self.known_names)} registered face(s) from database.")
            return True
        except Exception as e:
            logger.error(f"Database access error in face recognizer: {e}")
            return False

    def add_face(self, name, image_path, db_session):
        """
        Processes a face image, generates the 128-d encoding, and stores it.
        """
        if self.is_mock:
            # Simulated face encoding registration
            from backend.models import KnownFace
            mock_encoding = [float(np.random.normal(0, 0.1)) for _ in range(128)]
            
            # Check if name already exists
            existing = db_session.query(KnownFace).filter_by(name=name).first()
            if existing:
                existing.encoding_json = json.dumps(mock_encoding)
                existing.image_path = image_path
            else:
                new_face = KnownFace(
                    name=name,
                    encoding_json=json.dumps(mock_encoding),
                    image_path=image_path
                )
                db_session.add(new_face)
            
            db_session.commit()
            self.load_known_faces(db_session)
            return True
            
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) == 0:
                logger.error("No face detected in the provided image.")
                return False
                
            face_encoding = encodings[0] # Take first face
            
            from backend.models import KnownFace
            existing = db_session.query(KnownFace).filter_by(name=name).first()
            if existing:
                existing.encoding_json = json.dumps(face_encoding.tolist())
                existing.image_path = image_path
            else:
                new_face = KnownFace(
                    name=name,
                    encoding_json=json.dumps(face_encoding.tolist()),
                    image_path=image_path
                )
                db_session.add(new_face)
                
            db_session.commit()
            self.load_known_faces(db_session)
            logger.info(f"Registered face encoding for {name}")
            return True
        except Exception as e:
            logger.error(f"Error registering face: {e}")
            return False

    def recognize(self, frame):
        """
        Detects faces in frame and compares them to registered encodings.
        Returns a list of dicts: { name: str, bbox: [x1, y1, x2, y2] }
        """
        results = []
        
        if self.is_mock:
            return self._run_mock_recognition(frame)
            
        try:
            # Find face locations and encodings
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = "Unknown"
                
                if len(self.known_encodings) > 0:
                    matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=0.6)
                    face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = self.known_names[best_match_index]
                
                results.append({
                    "name": name,
                    "bbox": [left, top, right, bottom]
                })
        except Exception as e:
            logger.error(f"Error running face recognition: {e}")
            return self._run_mock_recognition(frame)
            
        return results

    def _run_mock_recognition(self, frame):
        """
        Simulate face recognition.
        If there are registered faces, maps the person in the simulated environment
        to one of the names.
        """
        # If no registered names, return empty list
        if not self.known_names:
            return []
            
        # Draw a face box inside the simulated person's head region
        # Simulated Person Center: (500, 200) -> Head: bbox [470, 170, 530, 230]
        # We can dynamically move this box with the person's coordinates
        import time
        t = time.time()
        offset = (int(t * 15) % 120)
        p_x1 = int(450 - offset // 2)
        p_y1 = int(230 - offset // 3)
        p_x2 = int(550 + offset // 2)
        
        # Estimate head coordinates from person bbox
        head_cx = (p_x1 + p_x2) // 2
        head_cy = p_y1
        
        # Face bounding box
        face_x1 = head_cx - 18
        face_y1 = head_cy - 20
        face_x2 = head_cx + 18
        face_y2 = head_cy + 15
        
        # Pick first registered name for mock demonstration
        mock_name = self.known_names[0] if len(self.known_names) > 0 else "Samantha"
        
        return [{
            "name": mock_name,
            "bbox": [face_x1, face_y1, face_x2, face_y2]
        }]
