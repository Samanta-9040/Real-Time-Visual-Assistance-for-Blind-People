import cv2
import base64
import time
import logging
import threading
import numpy as np
from datetime import datetime
from backend.config import (
    OCR_INTERVAL_FRAMES, SCENE_INTERVAL_FRAMES, DEPTH_INTERVAL_FRAMES,
    DEFAULT_FPS, FORCE_MOCK_AI, FRAME_WIDTH, FRAME_HEIGHT
)
from backend.ai_engine.camera_manager import CameraManager
from backend.ai_engine.object_detector import ObjectDetector
from backend.ai_engine.ocr_reader import OCRReader
from backend.ai_engine.scene_describer import SceneDescriber
from backend.ai_engine.depth_estimator import DepthEstimator
from backend.ai_engine.face_recognizer import FaceRecognizer
from backend.ai_engine.audio_engine import AudioEngine

from backend.utils.draw_utils import draw_detections, draw_ocr, draw_faces, blend_depth_map
from backend.utils.distance_calc import calculate_region

logger = logging.getLogger(__name__)

class FrameProcessor:
    def __init__(self, app, db, socketio, session_id, source_type="webcam", source_path=0):
        self.app = app
        self.db = db
        self.socketio = socketio
        self.session_id = session_id
        
        self.camera_manager = CameraManager(source_type, source_path, fps=DEFAULT_FPS)
        self.object_detector = ObjectDetector()
        self.ocr_reader = OCRReader()
        self.scene_describer = SceneDescriber()
        self.depth_estimator = DepthEstimator()
        self.face_recognizer = FaceRecognizer()
        self.audio_engine = AudioEngine()
        
        self.is_running = False
        self.thread = None
        self.frame_count = 0
        self.total_detections_count = 0
        
        self.latest_scene_description = ""
        self.latest_depth_map = None
        self.latest_ocr = []
        self.latest_faces = []
        
    def start(self):
        self.is_running = True
        
        # Load AI models
        self.object_detector.load_model()
        self.ocr_reader.load_model()
        self.scene_describer.load_model()
        self.depth_estimator.load_model()
        
        # Load face recognizer database encodings (using app context)
        with self.app.app_context():
            self.face_recognizer.load_known_faces(self.db.session)
            
        self.camera_manager.start()
        self.audio_engine.start()
        
        # Start execution loop thread
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info(f"FrameProcessor thread started for session {self.session_id}")

    def stop(self):
        self.is_running = False
        if self.camera_manager:
            self.camera_manager.stop()
        if self.audio_engine:
            self.audio_engine.stop()
            
        if self.thread:
            self.thread.join()
            
        # Update SessionLog end details
        with self.app.app_context():
            from backend.models import SessionLog
            session_record = SessionLog.query.filter_by(session_id=self.session_id).first()
            if session_record:
                session_record.end_time = datetime.utcnow()
                session_record.total_frames = self.frame_count
                session_record.total_detections = self.total_detections_count
                self.db.session.commit()
                
        logger.info(f"FrameProcessor thread stopped for session {self.session_id}")

    def _run_loop(self):
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = float(DEFAULT_FPS)
        
        for frame, metadata in self.camera_manager.get_frames():
            if not self.is_running:
                break
                
            self.frame_count += 1
            fps_frame_count += 1
            
            # Recalculate FPS every 2 seconds
            now = time.time()
            elapsed_fps = now - fps_start_time
            if elapsed_fps >= 2.0:
                current_fps = round(fps_frame_count / elapsed_fps, 1)
                fps_frame_count = 0
                fps_start_time = now
                
            # 1. Object Detection (Every frame)
            detections = self.object_detector.detect(frame)
            
            # 2. Depth Estimation (Every 5th frame)
            if self.frame_count % DEPTH_INTERVAL_FRAMES == 0 or self.latest_depth_map is None:
                self.latest_depth_map = self.depth_estimator.estimate(frame)
                
            # Cross-validate distances if depth map is ready
            if self.latest_depth_map is not None:
                detections = self.depth_estimator.cross_validate(self.latest_depth_map, detections)
                
            # 3. Face Recognition
            # Run face recognition if person detected, to conserve resources
            has_person = any(d["label"] == "person" for d in detections)
            if has_person or self.camera_manager.source_type == "mock":
                # Check face recognition every 5 frames to reduce load
                if self.frame_count % 5 == 0:
                    self.latest_faces = self.face_recognizer.recognize(frame)
            else:
                self.latest_faces = []

            # 4. OCR Reader (Every 10 frames)
            if self.frame_count % OCR_INTERVAL_FRAMES == 0:
                self.latest_ocr = self.ocr_reader.read_text(frame)
                
            # 5. Scene Describer (Every 30 frames or if significant change)
            should_describe = (self.frame_count % SCENE_INTERVAL_FRAMES == 0)
            if not should_describe and self.frame_count % 5 == 0:
                # Gated by frame difference changes
                should_describe = self.scene_describer.detect_significant_change(frame)
                
            if should_describe or not self.latest_scene_description:
                self.latest_scene_description = self.scene_describer.describe(frame)

            # 6. Compile Audio Speech Notifications
            self.audio_engine.format_and_speak(
                detections, self.latest_ocr, self.latest_scene_description, elapsed_fps
            )

            # 7. Draw Visual Annotations (Color bboxes, overlays)
            annotated_frame = frame.copy()
            
            # Draw depth heatmap if depth map exists (Blend it with alpha overlay)
            # (Toggleable on client, let's always apply it partially or send clean + depth frames)
            # Let's send blended depth map if settings.night_mode is active or overlay setting is on
            # We'll just draw bboxes, OCR and faces on a copy
            annotated_frame = draw_detections(annotated_frame, detections)
            annotated_frame = draw_ocr(annotated_frame, self.latest_ocr)
            annotated_frame = draw_faces(annotated_frame, self.latest_faces)
            
            # Draw blinking red live dot
            cv2.circle(annotated_frame, (25, 25), 6, (0, 0, 255), -1)
            cv2.putText(annotated_frame, "LIVE", (38, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"FPS: {current_fps}", (FRAME_WIDTH - 90, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 8. Encode to base64 JPEG for transmission
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_b64 = ""
            if ret:
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
            # Generate depth map image in base64 if needed by dashboard
            depth_b64 = ""
            if self.latest_depth_map is not None:
                depth_u8 = (self.latest_depth_map * 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
                ret_d, buf_d = cv2.imencode('.jpg', depth_colored)
                if ret_d:
                    depth_b64 = base64.b64encode(buf_d).decode('utf-8')

            # 9. Emit WebSocket Frame payload to connected clients
            payload = {
                "frame_b64": frame_b64,
                "depth_b64": depth_b64,
                "detections": detections,
                "ocr_results": self.latest_ocr,
                "scene_description": self.latest_scene_description,
                "faces": self.latest_faces,
                "fps": current_fps,
                "timestamp": metadata["timestamp"],
                "audio_queue_size": self.audio_engine.queue.qsize()
            }
            
            self.socketio.emit("frame_update", payload)

            # Emit immediate alarm warnings for CRITICAL objects inside 1.5m
            for det in detections:
                if det["priority"] == "CRITICAL" and det["distance_m"] <= 1.5:
                    alert_payload = {
                        "type": "CRITICAL",
                        "message": f"CRITICAL: {det['label']} directly ahead at {det['distance_m']}m!",
                        "object": det["label"],
                        "distance": det["distance_m"]
                    }
                    self.socketio.emit("alert", alert_payload)
                    
            # 10. Database Logging (using App Context in thread)
            self.total_detections_count += len(detections)
            with self.app.app_context():
                self._log_to_database(detections, self.latest_ocr, self.latest_faces)
                
    def _log_to_database(self, detections, ocr_results, faces):
        """
        Record detections and text logs in the database.
        """
        try:
            from backend.models import DetectionLog, TextLog, KnownFace
            import json
            
            # Log Detections
            for det in detections:
                log = DetectionLog(
                    session_id=self.session_id,
                    label=det["label"],
                    confidence=det["confidence"],
                    distance_m=det["distance_m"],
                    region=det["region"],
                    priority=det["priority"],
                    bbox_json=json.dumps(det["bbox"])
                )
                self.db.session.add(log)
                
            # Log OCR Hits
            for ocr in ocr_results:
                log = TextLog(
                    session_id=self.session_id,
                    text_found=ocr["text"],
                    category=ocr["category"],
                    confidence=ocr["confidence"],
                    region=ocr["region"]
                )
                self.db.session.add(log)
                
            # Increment Face recognition times
            for face in faces:
                if face["name"] != "Unknown":
                    db_face = KnownFace.query.filter_by(name=face["name"]).first()
                    if db_face:
                        db_face.times_recognized += 1
                        
            self.db.session.commit()
        except Exception as e:
            logger.error(f"Error logging to database: {e}")
            self.db.session.rollback()
