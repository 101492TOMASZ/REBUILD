"""
Script for evaluating BrandClassifier (MobileNetV2) accuracy
on a test dataset organized by brand folders.

Usage:
    python evaluate_classifier.py

Expected folder structure:
    test_classification/
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

# Add project root to path so we can import car_vision_app
sys.path.insert(0, str(Path(__file__).resolve().parent))

from car_vision_app.classification import BrandClassifier

# --- Configuration ---
MODEL_PATH = "car_detector_model/model.pth"
CLASSES_PATH = "car_detector_model/label_map.json"
TEST_DIR = Path("test_classification")
OUTPUT_CSV = "classification_results.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    # Verify test directory exists
    if not TEST_DIR.exists():
        print(f"ERROR: Test directory '{TEST_DIR}' not found.")
        sys.exit(1)

    # Load classifier
    print("Loading BrandClassifier...")
    classifier = BrandClassifier(MODEL_PATH, CLASSES_PATH)
    print()

    # Collect all images grouped by true brand (folder name)
    results = []
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

            # Run prediction
            pred_brand, confidence = classifier.predict(image)
            correct = pred_brand.upper() == true_brand

            results.append({
                "filename": img_path.name,
                "true_brand": true_brand,
                "predicted_brand": pred_brand,
                "confidence": round(confidence, 2),
                "correct": correct,
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

    # Per-brand statistics
    brand_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        brand_stats[r["true_brand"]]["total"] += 1
        if r["correct"]:
            brand_stats[r["true_brand"]]["correct"] += 1

    # --- Print summary ---
    print("\n" + "=" * 50)
    print("  CLASSIFICATION RESULTS")
    print("=" * 50)
    print(f"\n  Total images tested : {total}")
    print(f"  Correct predictions : {total_correct}")
    print(f"  Overall accuracy    : {overall_accuracy:.1f}%")
    print("\n  Per-brand accuracy:")
    print("  " + "-" * 40)

    for brand in sorted(brand_stats):
        s = brand_stats[brand]
        acc = s["correct"] / s["total"] * 100
        print(f"    {brand:<15} {s['correct']:>3}/{s['total']:<3}  ({acc:.1f}%)")

    print("  " + "-" * 40)
    print()


if __name__ == "__main__":
    main()
