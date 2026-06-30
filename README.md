# VisionBridge — Real-Time Visual Assistance for Blind People

VisionBridge is a full-stack, AI-powered assistive technology platform designed to give blind and visually impaired users a real-time understanding of their physical surroundings. 

The system leverages a camera feed to detect objects, read signs, estimate distances, describe scene layouts, recognize registered faces, and deliver spoken alerts via spatial browser audio — acting as an intelligent "seeing companion."

---

## 🏗️ System Architecture

```
                       VisionBridge Architecture
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  ┌──────────────┐    ┌───────────────┐    ┌─────────────┐  │
  │  │  Camera /    │───▶│  Python AI    │───▶│  Flask API  │  │
  │  │  Webcam /    │    │  Engine       │    │  REST +     │  │
  │  │  Mobile Cam  │    │  (YOLO +      │    │  WebSocket  │  │
  │  └──────────────┘    │  OCR + Depth) │    └──────┬──────┘  │
  │                      └───────────────┘           │         │
  │                                           ┌──────▼──────┐  │
  │                                           │  Frontend   │  │
  │                                           │  Dashboard  │  │
  │                                           │  + Audio UI │  │
  │                                           └─────────────┘  │
  └─────────────────────────────────────────────────────────────┘
```

- **Tier 1 — AI Engine (Python)**: Handles video processing, YOLOv8 object detection, EasyOCR/Tesseract text reading, MiDaS depth estimation, dlib face recognition, and audio alert prioritization.
- **Tier 2 — Backend API (Flask + SocketIO)**: Manages frames, serves REST endpoints, maintains SQLite session logs, and broadcasts WebSocket packets.
- **Tier 3 — Frontend (React 18)**: Hosts a caregiver dashboard with charts and logs under `/dashboard`, and an audio-first accessible interface with voice commands under `/user`.

---

## 🛠️ Prerequisites

- **Python**: Version 3.10+
- **Node.js**: Version 18+ (with npm)
- **Docker**: Optional (for containerized deployments)
- **Tesseract OCR Engine**: Required for the OCR fallback (if running natively).
  - *Windows*: Download from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki). Add the executable to your System PATH or set `pytesseract.pytesseract.tesseract_cmd` path.
  - *Ubuntu*: `sudo apt-get install tesseract-ocr`

---

## 🚀 Getting Started

### Option A: Run Natively (Recommended for Windows testing)

#### 1. Setup Backend
1. Open a terminal and navigate to `/backend`.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask server:
   ```bash
   python app.py
   ```
   *Note: On startup, if deep learning models or face recognition packages are missing, the backend will auto-switch to a simulated AI generator so the dashboard is fully functional out-of-the-box.*

#### 2. Setup Frontend
1. Open a new terminal and navigate to `/frontend`.
2. Install Node dependencies:
   ```bash
   npm install
   ```
3. Run the React developer server:
   ```bash
   npm run dev
   ```
4. Access the applications:
   - **Vite Hub**: `http://localhost:3000/`
   - **Caregiver Dashboard**: `http://localhost:3000/dashboard`
   - **Accessible Blind UI**: `http://localhost:3000/user`

---

### Option B: Run Containerized (Docker)

To run the entire multi-container stack (Nginx, React Frontend, Flask Backend) with one command:
```bash
docker-compose up --build
```
This serves:
- **VisionBridge Main Gateway**: `http://localhost/`
- **Caregiver Dashboard**: `http://localhost/dashboard`
- **Accessible Blind UI**: `http://localhost/user`

---

## 🤖 Model Download & Cache Instructions

When running in **Live AI Mode** (not simulated), the engine downloads weights automatically:
1. **YOLOv8**: `ultralytics` will download `yolov8n.pt` on first startup and save it to the `/models` directory.
2. **BLIP-2**: Hugging Face `transformers` will download and cache the `Salesforce/blip2-opt-2.7b` pipeline weights in your user home directory.
3. **MiDaS**: PyTorch Hub downloads and caches the model weights automatically on first run.

---

## ⚙️ Configuration Guide (`backend/config.py`)

Key parameters in `backend/config.py` include:
- `FORCE_MOCK_AI`: Set `True` to force simulation outputs (useful for headless testing).
- `DEFAULT_FPS`: Processing and streaming speed (default 10 FPS).
- `BLUR_THRESHOLD`: Variance score below which blurry frames are skipped (default 15.0).
- `CONFIDENCE_THRESHOLD`: Object detection threshold (default 0.45).
- `OCR_INTERVAL_FRAMES`: OCR runs every N frames (default 10) to reduce CPU load.
- `SCENE_INTERVAL_FRAMES`: Captioning runs every N frames (default 30).
- `DEFAULT_MODE`: Voice mode: `verbose` (speak all), `standard` (warning + OCR), `minimal` (hazards only).

---

## 📊 REST API & WebSockets

### REST API Endpoints
- `GET /api/settings`: Fetch current system parameters.
- `POST /api/settings`: Update settings (mode, threshold, tts speed, night mode).
- `GET /api/session/status`: Check if a camera capture thread is currently running.
- `POST /api/session/start`: Starts capturing. Takes source parameters:
  `{ "source": "webcam"|"ip_camera"|"mock", "ip_url": "..." }`
- `POST /api/session/stop`: Stop camera streaming.
- `GET /api/detections/history`: Query SQL logs.
- `POST /api/known_faces/add`: Register a new face profile (Multipart upload containing `name` and `image` file).
- `GET /api/known_faces/list`: Retrieve registered faces.

### WebSocket Events
- **Server Emits**:
  - `frame_update`: Emitted every 100ms containing base64 JPEG, detections list, OCR texts, scene caption, and audio queue size.
  - `alert`: Emitted immediately on warning conditions (critical obstacles within 1.5m).
- **Client Emits**:
  - `connect_camera`: Triggers backend frame grabber initialization.

---

## 🧪 Testing & Evaluation

Run the validation suite to confirm latency, OCR rates, and endpoint schemas:
1. **API Schema Checks**:
   ```bash
   python test_api_endpoints.py
   ```
2. **Object Detection Accuracy**:
   ```bash
   python test_detection_accuracy.py
   ```
3. **OCR CER/WER Benchmarks**:
   ```bash
   python test_ocr_accuracy.py
   ```
4. **Latency Verification**:
   ```bash
   python test_audio_latency.py
   ```
5. **System Resource Usage**:
   ```bash
   python performance_benchmark.py
   ```

---

## ✨ Unique Differentiators
1. **Simulation Fallback Mode**: Full stack runs flawlessly on any hardware even without GPUs or dlib installed.
2. **Spatial Audio Warning Cues**: Utilizes browser-side Web Audio API panning so hazard indicators sound physically left or right relative to the user.
3. **Centroid tracking**: Detects dynamic items moving towards the user and gives prompt warnings.
4. **Caregiver Log Replay**: Provides timeline charts and searchable SQL history of past detections.
