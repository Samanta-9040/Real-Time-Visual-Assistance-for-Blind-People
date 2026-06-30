import time
import logging
import threading
from queue import PriorityQueue

logger = logging.getLogger(__name__)

# Try importing pyttsx3
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logger.warning("pyttsx3 not installed. Checking for gTTS / pygame.")

# Try importing gTTS and pygame
try:
    from gtts import gTTS
    import pygame
    # Initialize pygame mixer
    pygame.mixer.init()
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    logger.warning("gtts or pygame not installed. Backend audio alerts will write to logs.")

class AudioEngine:
    def __init__(self, rate=160, volume=1.0, mode="standard"):
        self.rate = rate
        self.volume = volume
        self.mode = mode # verbose, standard, minimal
        self.queue = PriorityQueue()
        self.spoken_log = {} # message -> last_spoken_timestamp
        self.tts_engine = None
        self.is_running = False
        self.worker_thread = None
        self.last_no_detection_time = 0
        
    def init_tts(self):
        """
        Initializes offline TTS engine (pyttsx3) and falls back if unavailable.
        """
        if PYTTSX3_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', self.rate)
                self.tts_engine.setProperty('volume', self.volume)
                
                # Try setting female voice
                voices = self.tts_engine.getProperty('voices')
                for voice in voices:
                    if "female" in voice.name.lower() or "zira" in voice.name.lower() or "samantha" in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                logger.info("pyttsx3 TTS initialized successfully.")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize pyttsx3: {e}. Falling back to gTTS.")
                self.tts_engine = None
                
        if GTTS_AVAILABLE:
            logger.info("gTTS + pygame audio synthesis initialized as backup.")
            return True
            
        logger.warning("No backend audio synthesis tools available. Alerts will be logged to text only.")
        return True

    def start(self):
        self.is_running = True
        self.init_tts()
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        logger.info("Audio Engine queue processing started.")

    def stop(self):
        self.is_running = False
        # Inject dummy message to unblock queue.get()
        self.queue.put((99, "stop_signal", ""))
        if self.worker_thread:
            self.worker_thread.join()
        logger.info("Audio Engine queue processing stopped.")

    def change_settings(self, mode=None, rate=None, volume=None):
        if mode:
            self.mode = mode
        if rate:
            self.rate = rate
            if self.tts_engine:
                try:
                    self.tts_engine.setProperty('rate', self.rate)
                except Exception:
                    pass
        if volume:
            self.volume = volume
            if self.tts_engine:
                try:
                    self.tts_engine.setProperty('volume', self.volume)
                except Exception:
                    pass
        logger.info(f"Audio Settings updated: mode={self.mode}, rate={self.rate}, volume={self.volume}")

    def speak(self, text, priority=2, deduplicate_sec=0.0):
        """
        Add speech task to priority queue.
        Priorities:
        0 - Hazard alerts (Immediate)
        1 - General objects / text detections (High)
        2 - Environment scene descriptions / status updates (Normal)
        """
        if not text:
            return
            
        # Deduplication check
        now = time.time()
        if deduplicate_sec > 0.0:
            if text in self.spoken_log:
                if now - self.spoken_log[text] < deduplicate_sec:
                    # Skip duplicate speech
                    return
            self.spoken_log[text] = now
            
        # Put item in priority queue (item format: (priority, timestamp, text))
        # We add timestamp to keep queue sorting stable for matching priorities
        self.queue.put((priority, now, text))

    def _process_queue(self):
        while self.is_running:
            try:
                # Blocks until an item is available
                priority, timestamp, text = self.queue.get()
                
                if text == "stop_signal":
                    break
                    
                # Interrupt logic:
                # If a priority 0 (Hazard) message arrives while another speech is happening,
                # we want to abort current speech. Pyttsx3 does not natively support easy preemption 
                # in a single thread easily, but we can do a stop() check if speaking.
                
                logger.info(f"Speaking alert (P{priority}): {text}")
                
                self._synthesize_speech(text, priority)
                self.queue.task_done()
            except Exception as e:
                logger.error(f"Error in Audio Engine queue processor: {e}")
                time.sleep(0.1)

    def _synthesize_speech(self, text, priority):
        """
        Synthesizes text using available engines.
        """
        # If pyttsx3 is available
        if self.tts_engine:
            try:
                # Run the speech
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                return
            except Exception as e:
                logger.error(f"pyttsx3 speaking failed: {e}. Falling back.")
                
        # If gTTS is available and online
        if GTTS_AVAILABLE:
            try:
                import tempfile
                import os
                
                # Make speech mp3
                tts = gTTS(text=text, lang='en', slow=False)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    temp_filename = fp.name
                    
                tts.save(temp_filename)
                
                # Play with pygame
                pygame.mixer.music.load(temp_filename)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy() and self.is_running:
                    # Check for higher priority interrupts in queue
                    # (Simple implementation: if priority 0 alert arrives, stop playing backup)
                    if priority > 0 and not self.queue.empty():
                        next_p = self.queue.queue[0][0]
                        if next_p == 0:
                            pygame.mixer.music.stop()
                            logger.info("Speech playback interrupted by critical hazard.")
                            break
                    time.sleep(0.05)
                    
                # Clean up file
                pygame.mixer.music.unload()
                os.remove(temp_filename)
                return
            except Exception as e:
                logger.error(f"gTTS speaking failed: {e}")

        # Final fallback: log to console/server output
        print(f"[AUDIO OUTPUT]: {text}")

    def format_and_speak(self, detections, ocr_results, scene_desc, time_elapsed):
        """
        Formats detections based on current user mode and queues them.
        Modes:
        - VERBOSE: Speak everything
        - STANDARD: WARNING + CRITICAL + OCR
        - MINIMAL: CRITICAL only
        """
        now = time.time()
        
        # 1. Critical Hazards (Priority 0)
        critical_alerts = []
        for det in detections:
            if det["priority"] == "CRITICAL" and det["distance_m"] <= 1.5:
                msg = f"WARNING - {det['label']} directly ahead, {det['distance_m']} meters. Stop."
                self.speak(msg, priority=0, deduplicate_sec=3.0)
                critical_alerts.append(msg)
                
            elif det.get("is_approaching", False) and det["priority"] in ["CRITICAL", "WARNING"]:
                msg = f"Object {det['label']} approaching from the {det['region'].lower()}."
                self.speak(msg, priority=0, deduplicate_sec=4.0)
                critical_alerts.append(msg)
                
        # If we had hazards, they interrupt everything, so skip standard speech in this frame
        if len(critical_alerts) > 0:
            return

        # 2. General Object Detections (Priority 1)
        if self.mode in ["verbose", "standard"]:
            for det in detections:
                label = det["label"]
                dist = det["distance_m"]
                region = det["region"].lower()
                priority = det["priority"]
                
                # In standard mode, only speak CRITICAL (that aren't hazards yet) and WARNINGS
                if self.mode == "standard" and priority == "INFO":
                    continue
                    
                # Standard labeling speech format
                dist_str = f"{dist:.1f}".replace(".0", "")
                msg = f"{label} {dist_str} meters to your {region}."
                
                # Speak with deduplication of 4.0 seconds
                self.speak(msg, priority=1, deduplicate_sec=4.0)

        # 3. OCR results (Priority 1)
        if self.mode in ["verbose", "standard"]:
            for ocr in ocr_results:
                text = ocr["text"]
                category = ocr["category"]
                
                msg = f"Sign reads: {text}"
                if category == "price":
                    msg = f"Price reads: {text}"
                elif category == "number/address":
                    msg = f"Number reads: {text}"
                    
                self.speak(msg, priority=1, deduplicate_sec=10.0)

        # 4. Scene Descriptions (Priority 2)
        if self.mode == "verbose" and scene_desc:
            self.speak(scene_desc, priority=2, deduplicate_sec=20.0)

        # 5. Empty path announcement
        if len(detections) == 0 and len(ocr_results) == 0:
            if now - self.last_no_detection_time >= 5.0:
                self.speak("Path appears clear.", priority=2)
                self.last_no_detection_time = now
        else:
            self.last_no_detection_time = now
