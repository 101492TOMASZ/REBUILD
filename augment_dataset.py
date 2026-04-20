"""
Symulacja trudnych warunków na zdjęciach z dataset_german.
Tworzy folder dataset_german_augmented z kopiami zdjęć poddanymi augmentacjom.
Oryginalne zdjęcia pozostają nienaruszone.
"""

import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

# ─── konfiguracja ───────────────────────────────────────────────
SRC_DIR = Path("dataset_german")
DST_DIR = Path("dataset_german_augmented")
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─── funkcje augmentacji ────────────────────────────────────────

def darken(img: Image.Image) -> Image.Image:
    """Zmniejszenie jasności – symulacja nocy / ciemnego zdjęcia."""
    factor = random.uniform(0.2, 0.5)
    return ImageEnhance.Brightness(img).enhance(factor)


def brighten(img: Image.Image) -> Image.Image:
    """Zwiększenie jasności – symulacja przepalonego światła."""
    factor = random.uniform(1.8, 2.5)
    return ImageEnhance.Brightness(img).enhance(factor)


def gaussian_blur(img: Image.Image) -> Image.Image:
    """Rozmycie gaussowskie."""
    radius = random.uniform(2.0, 5.0)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def motion_blur(img: Image.Image) -> Image.Image:
    """Motion blur – rozmycie kierunkowe."""
    arr = np.array(img)
    size = random.choice([7, 11, 15])
    kernel = np.zeros((size, size))
    kernel[size // 2, :] = np.ones(size)
    # losowy kąt
    angle = random.uniform(0, 180)
    M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (size, size))
    kernel /= kernel.sum()
    blurred = cv2.filter2D(arr, -1, kernel)
    return Image.fromarray(blurred)


def add_noise(img: Image.Image) -> Image.Image:
    """Dodanie szumu gaussowskiego."""
    arr = np.array(img, dtype=np.float32)
    sigma = random.uniform(15, 40)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def jpeg_compression(img: Image.Image) -> Image.Image:
    """Symulacja silnej kompresji JPEG (utrata jakości)."""
    import io
    quality = random.randint(5, 20)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy()


def rotate(img: Image.Image) -> Image.Image:
    """Lekki obrót ±10–15 stopni."""
    angle = random.choice([-1, 1]) * random.uniform(10, 15)
    return img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))


def change_contrast(img: Image.Image) -> Image.Image:
    """Zmiana kontrastu – losowo niski lub wysoki."""
    factor = random.choice([random.uniform(0.3, 0.6), random.uniform(1.6, 2.5)])
    return ImageEnhance.Contrast(img).enhance(factor)


def partial_occlusion(img: Image.Image) -> Image.Image:
    """Częściowe zasłonięcie – losowy prostokąt imitujący przeszkodę."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    # prostokąt o rozmiarze 15-35% obrazu
    rw = random.randint(int(w * 0.15), int(w * 0.35))
    rh = random.randint(int(h * 0.15), int(h * 0.35))
    x = random.randint(0, max(w - rw, 0))
    y = random.randint(0, max(h - rh, 0))
    color = random.choice([(0, 0, 0), (128, 128, 128), (50, 50, 50)])
    draw.rectangle([x, y, x + rw, y + rh], fill=color)
    return img


def random_crop(img: Image.Image) -> Image.Image:
    """Ucięcie fragmentu auta – crop 60-80% oryginalnego rozmiaru."""
    w, h = img.size
    crop_ratio = random.uniform(0.6, 0.8)
    new_w = int(w * crop_ratio)
    new_h = int(h * crop_ratio)
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    cropped = img.crop((left, top, left + new_w, top + new_h))
    # przywróć oryginalny rozmiar
    return cropped.resize((w, h), Image.BICUBIC)


def rain_simulation(img: Image.Image) -> Image.Image:
    """Symulacja deszczu – losowe jasne linie + lekki blur + przyciemnienie."""
    arr = np.array(img, dtype=np.uint8).copy()
    h, w = arr.shape[:2]
    # krople deszczu
    num_drops = random.randint(300, 800)
    for _ in range(num_drops):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        length = random.randint(10, 30)
        thickness = 1
        x_end = x + random.randint(-2, 2)
        y_end = min(y + length, h - 1)
        cv2.line(arr, (x, y), (x_end, y_end), (200, 200, 200), thickness)
    # lekki blur + przyciemnienie
    arr = cv2.GaussianBlur(arr, (3, 3), 0)
    result = Image.fromarray(arr)
    result = ImageEnhance.Brightness(result).enhance(0.75)
    return result


def dirt_simulation(img: Image.Image) -> Image.Image:
    """Symulacja brudu – losowe plamy/kółka o ziemistych kolorach."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    num_spots = random.randint(15, 40)
    for _ in range(num_spots):
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        r = random.randint(5, max(min(w, h) // 10, 6))
        # ziemisty / brązowy kolor z przezroczystością
        color = (
            random.randint(40, 100),
            random.randint(30, 70),
            random.randint(10, 40),
        )
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    # lekki blur żeby plamy wyglądały naturalniej
    return img.filter(ImageFilter.GaussianBlur(radius=1.5))


# ─── rejestr augmentacji ────────────────────────────────────────

AUGMENTATIONS = {
    "dark": darken,
    "bright": brighten,
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "noise": add_noise,
    "jpeg_compression": jpeg_compression,
    "rotate": rotate,
    "contrast": change_contrast,
    "occlusion": partial_occlusion,
    "crop": random_crop,
    "rain": rain_simulation,
    "dirt": dirt_simulation,
}


# ─── główna pętla ───────────────────────────────────────────────

def main():
    src = SRC_DIR
    dst = DST_DIR

    if not src.exists():
        print(f"Nie znaleziono folderu źródłowego: {src}")
        return

    image_extensions = {".jpg", ".jpeg", ".png"}

    # zbierz pliki
    all_images = []
    for root, _dirs, files in os.walk(src):
        for fname in files:
            if Path(fname).suffix.lower() in image_extensions:
                all_images.append(Path(root) / fname)

    total = len(all_images)
    print(f"Znaleziono {total} zdjęć w {src}")

    done = 0
    errors = 0

    for img_path in all_images:
        rel = img_path.relative_to(src)  # np. AUDI/img001.jpg

        for aug_name, aug_fn in AUGMENTATIONS.items():
            # ścieżka docelowa: dataset_german_augmented/<aug_name>/<BRAND>/img.jpg
            out_path = dst / aug_name / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                img = Image.open(img_path).convert("RGB")
                augmented = aug_fn(img.copy())
                augmented.save(str(out_path), "JPEG", quality=90)
            except Exception as e:
                errors += 1
                print(f"  [BŁĄD] {aug_name} | {rel}: {e}")
                continue

        done += 1
        if done % 100 == 0 or done == total:
            print(f"  Przetworzono {done}/{total} zdjęć...")

    total_generated = done * len(AUGMENTATIONS) - errors
    print(f"\nGotowe! Wygenerowano {total_generated} augmentowanych zdjęć w '{dst}'")
    print(f"Augmentacje: {', '.join(AUGMENTATIONS.keys())}")
    if errors:
        print(f"Błędy: {errors}")


if __name__ == "__main__":
    main()
