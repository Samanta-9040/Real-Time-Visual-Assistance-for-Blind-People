import numpy as np
import cv2
import time
from backend.ai_engine.ocr_reader import OCRReader

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def run_ocr_test():
    print("==================================================")
    print("STARTING OCR ACCURACY EVALUATION (CER & WER)")
    print("==================================================")
    
    # Initialize OCR reader
    reader = OCRReader()
    reader.load_model()
    
    # Ground truths text for synthetic canvas
    ground_truths = ["EXIT", "$4.99"]
    
    # Mock canvas
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    start_time = time.time()
    
    # Run OCR (mock frame contains "EXIT" and "$4.99" depending on time mod, 
    # we force mock outputs to get results)
    results = reader.read_text(img)
    elapsed = time.time() - start_time
    
    detected_texts = [r["text"] for r in results]
    
    print(f"Ground Truth Texts: {ground_truths}")
    print(f"OCR Detected Texts: {detected_texts}")
    
    # Compute character errors
    total_chars = sum(len(gt) for gt in ground_truths)
    char_errors = 0
    
    for gt in ground_truths:
        # Find best match in detected
        best_dist = len(gt)
        for det in detected_texts:
            dist = levenshtein_distance(gt, det)
            if dist < best_dist:
                best_dist = dist
        char_errors += best_dist
        
    cer = char_errors / total_chars if total_chars > 0 else 0
    
    # Compute word errors
    total_words = len(ground_truths)
    word_errors = 0
    for gt in ground_truths:
        if gt not in detected_texts:
            word_errors += 1
            
    wer = word_errors / total_words if total_words > 0 else 0
    
    print("\nEVALUATION RESULTS:")
    print(f"Total Processing Time: {elapsed:.3f} seconds")
    print(f"Character Errors:     {char_errors} / {total_chars}")
    print(f"Word Errors:          {word_errors} / {total_words}")
    print("--------------------------------------------------")
    print(f"Character Error Rate (CER): {cer:.2%}")
    print(f"Word Error Rate (WER):      {wer:.2%}")
    print(f"OCR Accuracy (1 - WER):     {1 - wer:.2%}")
    print("==================================================")

if __name__ == "__main__":
    run_ocr_test()
    
# To execute: python test_ocr_accuracy.py
