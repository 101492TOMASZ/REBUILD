"""
Script for evaluating BrandClassifier (MobileNetV2) accuracy
on an augmented test dataset organized by augmentation type and brand folders.

Usage:
    python evaluate_classifier_augmented.py

Expected folder structure:
    dataset_german_augmented/
        bright/
            AUDI/
            BMW/
            ...
        contrast/
            AUDI/
            BMW/
            ...
        ...
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
TEST_DIR = Path("dataset_german_augmented")
OUTPUT_CSV = "classification_results_augmented.csv"
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

    results = []
    aug_dirs = sorted([d for d in TEST_DIR.iterdir() if d.is_dir()])

    if not aug_dirs:
        print(f"ERROR: No augmentation subdirectories found in '{TEST_DIR}'.")
        sys.exit(1)

    # Per-augmentation and per-brand stats
    aug_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    brand_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    aug_brand_stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "correct": 0}))

    for aug_dir in aug_dirs:
        aug_name = aug_dir.name
        brand_dirs = sorted([d for d in aug_dir.iterdir() if d.is_dir()])

        if not brand_dirs:
            print(f"  WARNING: No brand subdirectories in {aug_dir}")
            continue

        print(f"=== Augmentation: {aug_name} ===")

        for brand_dir in brand_dirs:
            true_brand = brand_dir.name.upper()
            images = sorted(
                f for f in brand_dir.iterdir()
                if f.suffix.lower() in IMAGE_EXTENSIONS
            )

            if not images:
                print(f"  WARNING: No images in {brand_dir}")
                continue

            print(f"  Processing {true_brand} ({len(images)} images)...")

            for img_path in images:
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"    WARNING: Could not read {img_path.name}, skipping.")
                    continue

                # Run prediction
                pred_brand, confidence = classifier.predict(image)
                correct = pred_brand.upper() == true_brand

                results.append({
                    "augmentation": aug_name,
                    "filename": img_path.name,
                    "true_brand": true_brand,
                    "predicted_brand": pred_brand,
                    "confidence": round(confidence, 2),
                    "correct": correct,
                })

                aug_stats[aug_name]["total"] += 1
                brand_stats[true_brand]["total"] += 1
                aug_brand_stats[aug_name][true_brand]["total"] += 1
                if correct:
                    aug_stats[aug_name]["correct"] += 1
                    brand_stats[true_brand]["correct"] += 1
                    aug_brand_stats[aug_name][true_brand]["correct"] += 1

    if not results:
        print("ERROR: No images were processed.")
        sys.exit(1)

    # --- Save results to CSV ---
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {OUTPUT_CSV}")

    # --- Compute overall metrics ---
    total = len(results)
    total_correct = sum(1 for r in results if r["correct"])
    overall_accuracy = total_correct / total * 100

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("  AUGMENTED CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"\n  Total images tested : {total}")
    print(f"  Correct predictions : {total_correct}")
    print(f"  Overall accuracy    : {overall_accuracy:.1f}%")

    # Per-augmentation accuracy
    print("\n  Per-augmentation accuracy:")
    print("  " + "-" * 50)
    for aug in sorted(aug_stats):
        s = aug_stats[aug]
        acc = s["correct"] / s["total"] * 100
        print(f"    {aug:<20} {s['correct']:>4}/{s['total']:<4}  ({acc:.1f}%)")
    print("  " + "-" * 50)

    # Per-brand accuracy (across all augmentations)
    print("\n  Per-brand accuracy (all augmentations combined):")
    print("  " + "-" * 50)
    for brand in sorted(brand_stats):
        s = brand_stats[brand]
        acc = s["correct"] / s["total"] * 100
        print(f"    {brand:<15} {s['correct']:>4}/{s['total']:<4}  ({acc:.1f}%)")
    print("  " + "-" * 50)

    # Detailed: per augmentation x brand
    print("\n  Detailed accuracy (augmentation x brand):")
    print("  " + "-" * 60)
    for aug in sorted(aug_brand_stats):
        print(f"    [{aug}]")
        for brand in sorted(aug_brand_stats[aug]):
            s = aug_brand_stats[aug][brand]
            acc = s["correct"] / s["total"] * 100
            print(f"      {brand:<15} {s['correct']:>3}/{s['total']:<3}  ({acc:.1f}%)")
    print("  " + "-" * 60)
    print()


if __name__ == "__main__":
    main()
