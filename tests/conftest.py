"""
Wspólne fixtures dla testów CarVision AI.
"""

import sys
import os
import pytest
import numpy as np
import cv2
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

# Dodaj katalog projektu do ścieżki importu
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

APP_DIR = PROJECT_ROOT / "car_vision_app"
MODEL_PATH = PROJECT_ROOT / "car_detector_model" / "model.pth"
CLASSES_PATH = PROJECT_ROOT / "car_detector_model" / "label_map.json"
YOLO_PATH = PROJECT_ROOT / "yolov8s.pt"
ANPR_MODEL_PATH = APP_DIR / "anpr_best.pt"


# ============================================================
# Obrazy syntetyczne
# ============================================================

@pytest.fixture
def black_image():
    """Czarny obraz 400x300."""
    return np.zeros((300, 400, 3), dtype=np.uint8)


@pytest.fixture
def white_image():
    """Biały obraz 400x300."""
    return np.ones((300, 400, 3), dtype=np.uint8) * 255


@pytest.fixture
def gray_image():
    """Szare tło z prostokątem imitującym auto."""
    img = np.full((480, 640, 3), 180, dtype=np.uint8)
    # Nadwozie
    cv2.rectangle(img, (100, 200), (540, 380), (60, 60, 90), -1)
    # Dach
    cv2.rectangle(img, (160, 130), (480, 220), (70, 70, 100), -1)
    return img


@pytest.fixture
def plate_image_plain():
    """Prosty biały obraz tablicy rejestracyjnej z tekstem."""
    img = np.ones((80, 350, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (2, 2), (348, 78), (0, 0, 0), 2)
    cv2.putText(img, "SC 6271X", (40, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    return img


@pytest.fixture
def plate_image_with_eu_strip():
    """Tablica z niebieskim pasem EU po lewej stronie."""
    img = np.ones((80, 350, 3), dtype=np.uint8) * 255
    # Niebieski pas EU (BGR: niebieski)
    eu_strip_w = 48
    img[:, :eu_strip_w] = [190, 20, 0]       # niebieski BGR
    # Żółte gwiazdki EU (symulacja)
    img[10:20, 14:26] = [0, 255, 255]
    # Tekst "PL"
    cv2.putText(img, "PL", (6, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    # Właściwy numer tablicy
    cv2.putText(img, "SC6271X", (60, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    return img


@pytest.fixture
def small_vehicle_detection():
    """Przykładowy słownik detekcji pojazdu (mała powierzchnia)."""
    return {
        'bbox': [50, 50, 150, 120],
        'score': 0.85,
        'class_id': 2,
        'area': 100 * 70,
        'center': (100.0, 85.0),
    }


@pytest.fixture
def large_vehicle_detection():
    """Przykładowy słownik detekcji pojazdu (duża powierzchnia, wycentrowany)."""
    return {
        'bbox': [80, 80, 560, 400],
        'score': 0.92,
        'class_id': 2,
        'area': 480 * 320,
        'center': (320.0, 240.0),
    }


# ============================================================
# Baza danych
# ============================================================

@pytest.fixture
def database(tmp_path):
    """
    Instancja Database z tymczasowym katalogiem —
    nie dotyka katalogu ~/.carvision użytkownika.
    """
    from car_vision_app.database import Database

    db = Database.__new__(Database)
    db.db_dir = tmp_path / "carvision_test"
    db.db_path = db.db_dir / "database.db"
    db.images_dir = db.db_dir / "images"
    db.db_dir.mkdir(exist_ok=True)
    db.images_dir.mkdir(exist_ok=True)
    db.conn = None
    db._connect()
    db._create_tables()

    yield db

    db.close()


# ============================================================
# ANPRModule bez modeli (do testów jednostkowych metod)
# ============================================================

@pytest.fixture
def anpr_bare():
    """
    ANPRModule z ustawionymi atrybutami ale bez załadowanych modeli YOLO/PaddleOCR.
    Używany do testowania metod czysto algorytmicznych.
    """
    from car_vision_app.anpr import ANPRModule

    with patch.object(ANPRModule, '__init__', return_value=None):
        module = ANPRModule()

    # Kopiuj atrybuty z prawdziwej klasy
    module.EU_COUNTRY_CODES = {
        'A', 'AL', 'AND', 'ARM', 'AZ', 'B', 'BG', 'BIH', 'BY', 'CH',
        'CY', 'CZ', 'D', 'DK', 'E', 'EST', 'F', 'FIN', 'FL', 'FR',
        'GB', 'GBG', 'GBJ', 'GBM', 'GBZ', 'GE', 'GR', 'H', 'HR',
        'HU', 'I', 'IRL', 'IS', 'KOS', 'L', 'LT', 'LV', 'M', 'MC',
        'MD', 'ME', 'MK', 'MNE', 'N', 'NL', 'NMK', 'P', 'PL', 'RKS',
        'RO', 'RSM', 'RUS', 'S', 'SK', 'SLO', 'SRB', 'TR', 'UA', 'V',
    }
    module.similar_chars = {
        '0': 'O', 'O': '0', '1': 'I', 'I': '1', 'L': '1',
        '2': 'Z', 'Z': '2', '4': 'A', 'A': '4', '5': 'S',
        'S': '5', '6': 'G', 'G': '6', '8': 'B', 'B': '8',
        'D': '0', 'Q': '0',
    }
    module.ocr_reader = MagicMock()
    module.plate_detector = MagicMock()

    return module


# ============================================================
# Ścieżki do rzeczywistych zasobów (pomijaj jeśli nie istnieją)
# ============================================================

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: wymaga prawdziwych plików modeli (wolne)"
    )
    config.addinivalue_line(
        "markers",
        "slow: testy zajmujące dużo czasu"
    )


requires_models = pytest.mark.skipif(
    not (MODEL_PATH.exists() and CLASSES_PATH.exists() and YOLO_PATH.exists()),
    reason="Wymaga plików modeli: model.pth, label_map.json, yolov8s.pt"
)

requires_anpr_model = pytest.mark.skipif(
    not ANPR_MODEL_PATH.exists(),
    reason=f"Wymaga modelu ANPR: {ANPR_MODEL_PATH}"
)
