import os
import uuid
import logging
import base64
from datetime import datetime
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from backend.config import SQLALCHEMY_DATABASE_URI, KNOWN_FACES_DIR, DEFAULT_FPS
from backend.models import db, SessionLog, DetectionLog, TextLog, KnownFace, Settings
from backend.ai_engine.frame_processor import FrameProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask and SocketIO
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

db.init_app(app)

# Global session tracker
active_processor = None
current_session_id = None

# Ensure tables and folders exist
with app.app_context():
    db.create_all()
    # Check if default settings exist, create if not
    default_settings = Settings.query.first()
    if not default_settings:
        default_settings = Settings(
            mode="standard",
            confidence_threshold=0.45,
            tts_rate=160,
            tts_volume=1.0,
            night_mode=False
        )
        db.session.add(default_settings)
        db.session.commit()

def stop_active_session():
    global active_processor, current_session_id
    if active_processor:
        logger.info(f"Stopping active session {current_session_id}...")
        active_processor.stop()
        active_processor = None
        current_session_id = None

# --- REST ENDPOINTS ---

@app.route('/api/session/start', methods=['POST'])
def start_session():
    global active_processor, current_session_id
    
    # Handle both multipart and json requests
    source_type = "webcam"
    source_path = 0
    
    if request.is_json:
        data = request.get_json() or {}
        source_type = data.get("source", "webcam")
        source_path = data.get("ip_url", 0)
    else:
        source_type = request.form.get("source", "webcam")
        source_path = request.form.get("ip_url", 0)
        
        # If a video file is uploaded
        if "file" in request.files and source_type == "upload":
            file = request.files["file"]
            if file.filename != '':
                upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
                os.makedirs(upload_dir, exist_ok=True)
                file_path = os.path.join(upload_dir, file.filename)
                file.save(file_path)
                source_path = file_path
                logger.info(f"Saved uploaded video to {file_path}")

    # Stop current session first
    stop_active_session()
    
    # Create new session DB record
    new_session_id = str(uuid.uuid4())
    current_session_id = new_session_id
    
    session_record = SessionLog(
        session_id=new_session_id,
        source_type=source_type
    )
    db.session.add(session_record)
    db.session.commit()
    
    # Load settings from db and apply to processor
    settings_record = Settings.query.first()
    
    # Initialize processor
    active_processor = FrameProcessor(
        app=app,
        db=db,
        socketio=socketio,
        session_id=new_session_id,
        source_type=source_type,
        source_path=source_path
    )
    
    # Configure initial processor settings
    if settings_record:
        active_processor.audio_engine.change_settings(
            mode=settings_record.mode,
            rate=settings_record.tts_rate,
            volume=settings_record.tts_volume
        )
        active_processor.camera_manager.night_mode = settings_record.night_mode
        active_processor.object_detector.is_mock = active_processor.object_detector.is_mock or settings_record.night_mode # (Force simulation if selected or just use YOLO settings)
        
    active_processor.start()
    
    return jsonify({
        "session_id": new_session_id,
        "status": "started"
    })

@app.route('/api/session/stop', methods=['POST'])
def stop_session():
    global active_processor, current_session_id
    
    if not active_processor:
        return jsonify({"error": "No active session to stop"}), 400
        
    session_id = current_session_id
    total_frames = active_processor.frame_count
    total_detections = active_processor.total_detections_count
    
    # Stop session
    stop_active_session()
    
    # Query session summary
    session_record = SessionLog.query.filter_by(session_id=session_id).first()
    duration = 0.0
    if session_record and session_record.end_time and session_record.start_time:
        duration = (session_record.end_time - session_record.start_time).total_seconds()
        
    summary = f"Session completed. Processed {total_frames} frames over {duration:.1f} seconds, with {total_detections} objects detected."
    
    return jsonify({
        "session_id": session_id,
        "total_detections": total_detections,
        "duration_seconds": duration,
        "summary": summary
    })

@app.route('/api/session/status', methods=['GET'])
def get_session_status():
    global active_processor
    if active_processor and active_processor.is_running:
        return jsonify({
            "active": True,
            "session_id": current_session_id,
            "fps": getattr(active_processor, 'current_fps', DEFAULT_FPS),
            "frame_count": active_processor.frame_count,
            "detections_this_session": active_processor.total_detections_count
        })
    return jsonify({"active": False})

@app.route('/api/detections/history', methods=['GET'])
def get_detections_history():
    session_id = request.args.get("session_id")
    limit = int(request.args.get("limit", 50))
    
    query = DetectionLog.query
    if session_id:
        query = query.filter_by(session_id=session_id)
        
    logs = query.order_by(DetectionLog.timestamp.desc()).limit(limit).all()
    return jsonify([log.to_dict() for log in logs])

@app.route('/api/stats/summary', methods=['GET'])
def get_stats_summary():
    # Number of sessions
    total_sessions = SessionLog.query.count()
    
    # Average duration
    sessions = SessionLog.query.filter(SessionLog.end_time.isnot(None)).all()
    avg_duration = 0.0
    if len(sessions) > 0:
        total_dur = sum((s.end_time - s.start_time).total_seconds() for s in sessions)
        avg_duration = total_dur / len(sessions)
        
    # Hazard counts (CRITICAL detections within 1.5 meters)
    hazard_count = DetectionLog.query.filter(
        DetectionLog.priority == "CRITICAL",
        DetectionLog.distance_m <= 1.5
    ).count()
    
    # Most detected objects
    # Query to count occurrences grouped by label
    results = db.session.query(
        DetectionLog.label, db.func.count(DetectionLog.label)
    ).group_by(DetectionLog.label).order_by(db.func.count(DetectionLog.label).desc()).limit(5).all()
    
    most_detected = [{"label": r[0], "count": r[1]} for r in results]
    
    return jsonify({
        "total_sessions": total_sessions,
        "avg_session_duration": avg_duration,
        "hazard_count": hazard_count,
        "most_detected_objects": most_detected
    })

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    settings_record = Settings.query.first()
    if not settings_record:
        settings_record = Settings()
        db.session.add(settings_record)
        db.session.commit()
        
    if request.method == 'POST':
        data = request.get_json() or {}
        settings_record.mode = data.get("mode", settings_record.mode)
        settings_record.confidence_threshold = float(data.get("confidence_threshold", settings_record.confidence_threshold))
        settings_record.tts_rate = int(data.get("tts_rate", settings_record.tts_rate))
        settings_record.tts_volume = float(data.get("tts_volume", settings_record.tts_volume))
        settings_record.night_mode = bool(data.get("night_mode", settings_record.night_mode))
        db.session.commit()
        
        # Apply changes live if processor is running
        global active_processor
        if active_processor:
            active_processor.audio_engine.change_settings(
                mode=settings_record.mode,
                rate=settings_record.tts_rate,
                volume=settings_record.tts_volume
            )
            active_processor.camera_manager.night_mode = settings_record.night_mode
            
        return jsonify(settings_record.to_dict())
        
    # GET request
    return jsonify(settings_record.to_dict())

@app.route('/api/known_faces/add', methods=['POST'])
def add_known_face():
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({"error": "Name and image file are required"}), 400
        
    name = request.form['name']
    image_file = request.files['image']
    
    if image_file.filename == '':
        return jsonify({"error": "No selected image file"}), 400
        
    # Save image to folder
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    file_ext = os.path.splitext(image_file.filename)[1]
    safe_name = "".join(x for x in name if x.isalnum() or x in " -_").strip()
    image_path = os.path.join(KNOWN_FACES_DIR, f"{safe_name}{file_ext}")
    image_file.save(image_path)
    
    # Process and register face encoding
    # We create a temporary recognizer to generate the vector and save it to the DB
    from backend.ai_engine.face_recognizer import FaceRecognizer
    recognizer = FaceRecognizer()
    success = recognizer.add_face(name, image_path, db.session)
    
    # If a live frame processor is active, notify it to reload encodings
    global active_processor
    if active_processor:
        active_processor.face_recognizer.load_known_faces(db.session)
        
    if success:
        return jsonify({"message": f"Successfully registered face for {name}"})
    else:
        if os.path.exists(image_path):
            os.remove(image_path)
        return jsonify({"error": "Could not register face encoding. Check if a clear face exists in the image."}), 400

@app.route('/api/known_faces/list', methods=['GET'])
def list_known_faces():
    faces = KnownFace.query.order_by(KnownFace.added_at.desc()).all()
    # Add base64 image data to response if possible, or just the paths
    results = []
    for f in faces:
        record = f.to_dict()
        # Convert image to base64 for direct dashboard display
        if f.image_path and os.path.exists(f.image_path):
            try:
                with open(f.image_path, "rb") as img_f:
                    record["image_base64"] = base64.b64encode(img_f.read()).decode('utf-8')
            except Exception:
                record["image_base64"] = ""
        else:
            record["image_base64"] = ""
        results.append(record)
    return jsonify(results)

# --- WEBSOCKET EVENT HANDLERS ---

@socketio.on('connect')
def handle_connect():
    logger.info(f"Websocket client connected: {request.sid}")
    emit('connection_response', {'status': 'connected', 'session_id': current_session_id})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Websocket client disconnected: {request.sid}")

@socketio.on('connect_camera')
def handle_connect_camera(data):
    """
    WebSocket client triggers camera streaming start.
    Payload: { "source": "webcam"|"mock"|..., "ip_url": "..." }
    """
    logger.info(f"WebSocket client requested camera connect: {data}")
    source_type = data.get("source", "webcam")
    ip_url = data.get("ip_url", 0)
    
    # Stop current session and start new
    global active_processor, current_session_id
    stop_active_session()
    
    new_session_id = str(uuid.uuid4())
    current_session_id = new_session_id
    
    # Record to DB
    session_record = SessionLog(
        session_id=new_session_id,
        source_type=source_type
    )
    db.session.add(session_record)
    db.session.commit()
    
    # Create processor
    settings_record = Settings.query.first()
    active_processor = FrameProcessor(
        app=app,
        db=db,
        socketio=socketio,
        session_id=new_session_id,
        source_type=source_type,
        source_path=ip_url
    )
    
    if settings_record:
        active_processor.audio_engine.change_settings(
            mode=settings_record.mode,
            rate=settings_record.tts_rate,
            volume=settings_record.tts_volume
        )
        active_processor.camera_manager.night_mode = settings_record.night_mode
        active_processor.object_detector.is_mock = active_processor.object_detector.is_mock or settings_record.night_mode
        
    active_processor.start()
    
    emit('camera_status', {'status': 'connected', 'session_id': new_session_id})

if __name__ == '__main__':
    # Start web server
    logger.info("Starting VisionBridge Flask server on port 5000...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
