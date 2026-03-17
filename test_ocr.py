#!/usr/bin/env python
"""Test EasyOCR functionality."""

import sys
import os
sys.path.insert(0, '/home/tomasz/Dokumenty/REBUILD')

print("Starting test...", flush=True)

try:
    print("Importing easyocr...", flush=True)
    import easyocr
    print("✓ EasyOCR imported", flush=True)
    
    print("Initializing reader...", flush=True)
    reader = easyocr.Reader(['en'], gpu=False)
    print("✓ EasyOCR reader initialized", flush=True)
    
    # Test z prostym tekstem
    import numpy as np
    import cv2
    
    test_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    cv2.putText(test_img, 'WOB G804', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    print("Running OCR on test image...", flush=True)
    result = reader.readtext(test_img)
    print(f"✓ OCR result: {result}", flush=True)
    
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
