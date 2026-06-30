import json
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class SessionLog(db.Model):
    __tablename__ = 'session_logs'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), unique=True, nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime, nullable=True)
    total_frames = db.Column(db.Integer, default=0)
    total_detections = db.Column(db.Integer, default=0)
    source_type = db.Column(db.String(50), nullable=False) # webcam, ip_camera, upload, mock

    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_frames': self.total_frames,
            'total_detections': self.total_detections,
            'source_type': self.source_type
        }

class DetectionLog(db.Model):
    __tablename__ = 'detection_logs'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), db.ForeignKey('session_logs.session_id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    label = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    distance_m = db.Column(db.Float, nullable=False)
    region = db.Column(db.String(20), nullable=False) # LEFT, CENTER, RIGHT
    priority = db.Column(db.String(20), nullable=False) # CRITICAL, WARNING, INFO
    bbox_json = db.Column(db.String(500), nullable=False) # JSON encoded [x1, y1, x2, y2]

    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'label': self.label,
            'confidence': self.confidence,
            'distance_m': self.distance_m,
            'region': self.region,
            'priority': self.priority,
            'bbox': json.loads(self.bbox_json) if self.bbox_json else []
        }

class TextLog(db.Model):
    __tablename__ = 'text_logs'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), db.ForeignKey('session_logs.session_id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    text_found = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50), nullable=False) # signage, number/address, price, general text
    confidence = db.Column(db.Float, nullable=False)
    region = db.Column(db.String(20), nullable=False) # LEFT, CENTER, RIGHT

    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'text_found': self.text_found,
            'category': self.category,
            'confidence': self.confidence,
            'region': self.region
        }

class KnownFace(db.Model):
    __tablename__ = 'known_faces'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    encoding_json = db.Column(db.Text, nullable=False) # JSON encoded 128-d face encoding vector
    image_path = db.Column(db.String(255), nullable=True)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)
    times_recognized = db.Column(db.Integer, default=0)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'image_path': self.image_path,
            'added_at': self.added_at.isoformat() if self.added_at else None,
            'times_recognized': self.times_recognized
        }

class Settings(db.Model):
    __tablename__ = 'settings'
    id = db.Column(db.Integer, primary_key=True)
    mode = db.Column(db.String(20), default='standard') # verbose, standard, minimal
    confidence_threshold = db.Column(db.Float, default=0.45)
    tts_rate = db.Column(db.Integer, default=160)
    tts_volume = db.Column(db.Float, default=1.0)
    night_mode = db.Column(db.Boolean, default=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'mode': self.mode,
            'confidence_threshold': self.confidence_threshold,
            'tts_rate': self.tts_rate,
            'tts_volume': self.tts_volume,
            'night_mode': self.night_mode,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
