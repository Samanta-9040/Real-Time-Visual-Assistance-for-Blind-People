import time
import logging
from backend.ai_engine.audio_engine import AudioEngine

def test_latency():
    print("==================================================")
    print("STARTING AUDIO ENGINE LATENCY BENCHMARK")
    print("==================================================")
    
    # Configure logging to silence standard logs during test
    logging.getLogger('backend.ai_engine.audio_engine').setLevel(logging.WARNING)
    
    # Initialize audio engine
    engine = AudioEngine(mode="verbose")
    engine.start()
    
    # Create latencies placeholder
    latencies = []
    
    # Patch the synthesis function to measure trigger differences
    original_speak = engine._synthesize_speech
    
    def patched_speak(text, priority):
        end_time = time.time()
        # Find corresponding start time from spoken log
        # Spoken log holds message -> timestamp
        start_time = engine.spoken_log.get(text, 0.0)
        if start_time > 0.0:
            latency = (end_time - start_time) * 1000.0 # ms
            latencies.append(latency)
            print(f"Triggered: '{text}' | Latency: {latency:.1f}ms")
        
        # Fast mock to avoid speaking audibly during benchmark
        time.sleep(0.01)
        
    engine._synthesize_speech = patched_speak
    
    # Test cases with different priority levels
    test_alerts = [
        ("WARNING - chair directly ahead, 1.2 meters. Stop.", 0), # P0 hazard
        ("person 2.0 meters to your left.", 1),                   # P1 detection
        ("You are in an indoor environment.", 2),                  # P2 scene
        ("WARNING - vehicle approaching from right.", 0),        # P0 hazard
        ("Sign reads: EXIT.", 1)                                   # P1 text
    ]
    
    print("Simulating audio triggers...")
    for text, priority in test_alerts:
        # Record trigger start time
        now = time.time()
        engine.spoken_log[text] = now
        
        # Push to queue
        engine.speak(text, priority=priority)
        time.sleep(0.2) # Small gap between events
        
    # Wait for queue to clear
    time.sleep(0.5)
    engine.stop()
    
    if len(latencies) > 0:
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        
        print("--------------------------------------------------")
        print(f"Total Triggers Measured: {len(latencies)}")
        print(f"Minimum Latency:         {min_latency:.1f}ms")
        print(f"Maximum Latency:         {max_latency:.1f}ms")
        print(f"Average Latency:         {avg_latency:.1f}ms")
        print("--------------------------------------------------")
        if avg_latency < 500.0:
            print("STATUS: PASS (Average latency is under the 500ms target)")
        else:
            print("STATUS: FAIL (Average latency exceeds the 500ms target)")
    else:
        print("Error: No latency measurements recorded.")
    print("==================================================")

if __name__ == "__main__":
    test_latency()
    
# To execute: python test_audio_latency.py
