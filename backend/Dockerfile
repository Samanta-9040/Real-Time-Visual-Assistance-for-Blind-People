# Dockerfile for backend
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    espeak \
    tesseract-ocr \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose backend port
EXPOSE 5000

# Start Flask gunicorn with eventlet class for websockets
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]
