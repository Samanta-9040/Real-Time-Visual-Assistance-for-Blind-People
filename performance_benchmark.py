import time
import numpy as np
import cv2
import psutil
import logging
from backend.ai_engine.object_detector import ObjectDetector
from backend.ai_engine.ocr_reader import OCRReader
from backend.ai_engine.scene_describer import SceneDescriber
from backend.ai_engine.depth_estimator import DepthEstimator

# Disable standard logger outputs during benchmark
logging.basicConfig(level=logging.ERROR)

def benchmark_modules():
    print("==================================================")
    print("STARTING VISIONBRIDGE PERFORMANCE & RESOURCE BENCHMARK")
    print("==================================================")
    
    # Track baseline resources
    process = psutil.Process()
    baseline_cpu = psutil.cpu_percent(interval=0.2)
    baseline_mem = process.memory_info().rss / (1024 * 1024) # MB
    
    # 1. Initialize Modules
    print("Initializing AI engine modules...")
    
    t_start = time.time()
    detector = ObjectDetector()
    detector.load_model()
    t_det = time.time() - t_start
    
    t_start = time.time()
    ocr = OCRReader()
    ocr.load_model()
    t_ocr = time.time() - t_start
    
    t_start = time.time()
    describer = SceneDescriber()
    describer.load_model()
    t_desc = time.time() - t_start
    
    t_start = time.time()
    depth = DepthEstimator()
    depth.load_model()
    t_depth = time.time() - t_start
    
    print("\nINITIALIZATION TIMES:")
    print(f"Object Detector:   {t_det:.2f}s (Mock: {detector.is_mock})")
    print(f"OCR Reader:        {t_ocr:.2f}s (Mock: {ocr.is_mock})")
    print(f"Scene Describer:   {t_desc:.2f}s (Mock: {describer.is_mock})")
    print(f"Depth Estimator:   {t_depth:.2f}s (Mock: {depth.is_mock})")
    print("--------------------------------------------------")
    
    # Create test dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 2. Benchmark Object Detector
    print("Benchmarking Object Detector (10 iterations)...")
    t0 = time.time()
    for _ in range(10):
        _ = detector.detect(frame)
    el_detector = (time.time() - t0) / 10.0
    
    # 3. Benchmark OCR
    print("Benchmarking OCR Reader (10 iterations)...")
    t0 = time.time()
    for _ in range(10):
        _ = ocr.read_text(frame)
    el_ocr = (time.time() - t0) / 10.0
    
    # 4. Benchmark Scene Describer
    print("Benchmarking Scene Describer (5 iterations)...")
    t0 = time.time()
    for _ in range(5):
        _ = describer.describe(frame)
    el_describer = (time.time() - t0) / 5.0
    
    # 5. Benchmark Depth Estimator
    print("Benchmarking Depth Estimator (10 iterations)...")
    t0 = time.time()
    for _ in range(10):
        _ = depth.estimate(frame)
    el_depth = (time.time() - t0) / 10.0

    # Read active resources
    active_cpu = psutil.cpu_percent(interval=0.2)
    active_mem = process.memory_info().rss / (1024 * 1024) # MB

    print("\nBENCHMARK RESULTS PER FRAME:")
    print(f"{'Module':<20} | {'Latency (ms)':<15} | {'Throughput (FPS)':<15}")
    print("-" * 56)
    print(f"{'Object Detector':<20} | {el_detector*1000.0:9.1f} ms | {1.0/el_detector:9.1f} FPS")
    print(f"{'OCR Reader':<20} | {el_ocr*1000.0:9.1f} ms | {1.0/el_ocr:9.1f} FPS")
    print(f"{'Scene Describer':<20} | {el_describer*1000.0:9.1f} ms | {1.0/el_describer:9.1f} FPS")
    print(f"{'Depth Estimator':<20} | {el_depth*1000.0:9.1f} ms | {1.0/el_depth:9.1f} FPS")
    print("--------------------------------------------------")
    
    # Calculate Gated Pipeline FPS (incorporating duty cycles: YOLO 1.0, Depth 0.2, OCR 0.1, Scene 0.033)
    pipeline_latency = (
        1.0 * el_detector +
        0.2 * el_depth +
        0.1 * el_ocr +
        0.033 * el_describer
    )
    pipeline_fps = 1.0 / pipeline_latency
    
    print(f"Estimated Gated Pipeline Latency: {pipeline_latency*1000.0:.1f}ms")
    print(f"Estimated Gated Pipeline FPS:     {pipeline_fps:.1f} FPS")
    print("--------------------------------------------------")
    print("RESOURCE CONSUMPTION:")
    print(f"CPU Utilization: {active_cpu:.1f}% (Baseline: {baseline_cpu:.1f}%)")
    print(f"Memory (RAM):    {active_mem:.1f} MB (Baseline: {baseline_mem:.1f} MB | Delta: {active_mem-baseline_mem:.1f} MB)")
    
    # Identify bottleneck
    latencies = {
        "YOLO Object Detector": el_detector,
        "EasyOCR Text Reader": el_ocr,
        "BLIP-2 Scene Describer": el_describer,
        "MiDaS Depth Estimator": el_depth
    }
    slowest_name = max(latencies, key=latencies.get)
    print(f"Bottleneck Identified: {slowest_name} ({latencies[slowest_name]*1000.0:.1f}ms / frame)")
    print("==================================================")

if __name__ == "__main__":
    benchmark_modules()
    
# To execute: python performance_benchmark.py
