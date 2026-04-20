"""
Script for evaluating BrandClassifier (MobileNetV2) accuracy
with a detection step: first crop the car using YOLOv8 CarDetector,
then classify the cropped image.

Usage:
    python evaluate_with_detection.py

Expected folder structure:
    dataset_german/
        AUDI/
        BMW/
        MERCEDES/
        PORSCHE/
        VOLKSWAGEN/
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from car_vision_app.detection import CarDetector
from car_vision_app.classification import BrandClassifier

# --- Configuration ---
YOLO_MODEL_PATH = "yolov8s.pt"
MODEL_PATH = "car_detector_model/model.pth"
CLASSES_PATH = "car_detector_model/label_map.json"
TEST_DIR = Path("dataset_german")
OUTPUT_CSV = "classification_results_with_detection.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    if not TEST_DIR.exists():
        print(f"ERROR: Test directory '{TEST_DIR}' not found.")
        sys.exit(1)

    print("Loading CarDetector (YOLOv8)...")
    detector = CarDetector(YOLO_MODEL_PATH)

    print("Loading BrandClassifier (MobileNetV2)...")
    classifier = BrandClassifier(MODEL_PATH, CLASSES_PATH)
    print()

    results = []
    no_detection_count = 0
    brand_dirs = sorted([d for d in TEST_DIR.iterdir() if d.is_dir()])

    if not brand_dirs:
        print(f"ERROR: No brand subdirectories found in '{TEST_DIR}'.")
        sys.exit(1)

    for brand_dir in brand_dirs:
        true_brand = brand_dir.name.upper()
        images = sorted(
            f for f in brand_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not images:
            print(f"  WARNING: No images in {brand_dir}")
            continue

        print(f"Processing {true_brand} ({len(images)} images)...")

        for img_path in images:
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  WARNING: Could not read {img_path.name}, skipping.")
                continue

            # Step 1: Detect and crop the car
            crop, vehicle_info, _ = detector.detect_and_crop(image)

            if crop is None:
                # No car detected – classify the full image as fallback
                no_detection_count += 1
                crop_used = image
                detected = False
            else:
                crop_used = crop
                detected = True

            # Step 2: Classify the cropped (or full) image
            pred_brand, confidence = classifier.predict(crop_used)
            correct = pred_brand.upper() == true_brand

            results.append({
                "filename": img_path.name,
                "true_brand": true_brand,
                "predicted_brand": pred_brand,
                "confidence": round(confidence, 2),
                "correct": correct,
                "car_detected": detected,
            })

    if not results:
        print("ERROR: No images were processed.")
        sys.exit(1)

    # --- Save results to CSV ---
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {OUTPUT_CSV}")

    # --- Compute metrics ---
    total = len(results)
    total_correct = sum(1 for r in results if r["correct"])
    overall_accuracy = total_correct / total * 100
    detected_count = sum(1 for r in results if r["car_detected"])

    # Per-brand statistics
    brand_stats = defaultdict(lambda: {"total": 0, "correct": 0, "detected": 0})
    for r in results:
        brand_stats[r["true_brand"]]["total"] += 1
        if r["correct"]:
            brand_stats[r["true_brand"]]["correct"] += 1
        if r["car_detected"]:
            brand_stats[r["true_brand"]]["detected"] += 1

    # --- Print summary ---
    print("\n" + "=" * 55)
    print("  CLASSIFICATION RESULTS (with detection crop)")
    print("=" * 55)
    print(f"\n  Total images tested : {total}")
    print(f"  Car detected        : {detected_count} ({detected_count / total * 100:.1f}%)")
    print(f"  No detection (full) : {no_detection_count}")
    print(f"  Correct predictions : {total_correct}")
    print(f"  Overall accuracy    : {overall_accuracy:.1f}%")
    print("\n  Per-brand accuracy:")
    print("  " + "-" * 48)

    for brand in sorted(brand_stats):
        s = brand_stats[brand]
        acc = s["correct"] / s["total"] * 100
        det = s["detected"] / s["total"] * 100
        print(f"    {brand:<15} {s['correct']:>3}/{s['total']:<3}  ({acc:.1f}%)  det: {det:.0f}%")

    print("  " + "-" * 48)
    print()


if __name__ == "__main__":
    main()
