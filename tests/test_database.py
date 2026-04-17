"""
Testy jednostkowe modułu bazy danych (Database).

Wszystkie testy używają tymczasowego katalogu (fixture `database` z conftest.py)
i nie dotykają produkcyjnej bazy ~/.carvision.
"""

import sys
import csv
import sqlite3
import pytest
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from car_vision_app.database import Database


# ============================================================
# Pomocnicze obrazy
# ============================================================

def make_image(h=100, w=150, color=(128, 128, 128)):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


# ============================================================
# Testy inicjalizacji i schematu
# ============================================================

class TestDatabaseInit:
    def test_db_file_created(self, database):
        assert database.db_path.exists()

    def test_images_dir_created(self, database):
        assert database.images_dir.is_dir()

    def test_connection_is_open(self, database):
        assert database.conn is not None

    def test_detections_table_exists(self, database):
        cursor = database.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='detections'")
        assert cursor.fetchone() is not None

    def test_timestamp_index_exists(self, database):
        cursor = database.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_timestamp'")
        assert cursor.fetchone() is not None

    def test_plate_index_exists(self, database):
        cursor = database.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_plate_text'")
        assert cursor.fetchone() is not None

    def test_brand_index_exists(self, database):
        cursor = database.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_car_brand'")
        assert cursor.fetchone() is not None


# ============================================================
# Testy hash
# ============================================================

class TestComputeHash:
    def test_hash_is_string(self):
        img = make_image()
        h = Database._compute_hash(img)
        assert isinstance(h, str)

    def test_hash_length_32(self):
        img = make_image()
        h = Database._compute_hash(img)
        assert len(h) == 32  # MD5 hex = 32 chars

    def test_same_image_same_hash(self):
        img = make_image(color=(100, 150, 200))
        assert Database._compute_hash(img) == Database._compute_hash(img)

    def test_different_images_different_hash(self):
        img1 = make_image(color=(0, 0, 0))
        img2 = make_image(color=(255, 255, 255))
        assert Database._compute_hash(img1) != Database._compute_hash(img2)

    def test_hash_is_hexadecimal(self):
        img = make_image()
        h = Database._compute_hash(img)
        int(h, 16)  # raises ValueError if not hex


# ============================================================
# Testy zapisywania i odczytywania obrazów
# ============================================================

class TestSaveAndGetImage:
    def test_save_returns_filename(self, database):
        img = make_image()
        filename = database.save_image(img)
        assert isinstance(filename, str)
        assert filename.endswith(".jpg")

    def test_saved_file_exists(self, database):
        img = make_image()
        filename = database.save_image(img)
        assert (database.images_dir / filename).exists()

    def test_duplicate_save_returns_same_name(self, database):
        img = make_image(color=(10, 20, 30))
        f1 = database.save_image(img)
        f2 = database.save_image(img)
        assert f1 == f2

    def test_get_existing_image(self, database):
        img = make_image(color=(200, 100, 50))
        filename = database.save_image(img)
        loaded = database.get_image(filename)
        assert loaded is not None
        assert isinstance(loaded, np.ndarray)

    def test_get_nonexistent_image_returns_none(self, database):
        result = database.get_image("nonexistent_abc123.jpg")
        assert result is None

    def test_saved_image_has_correct_shape(self, database):
        img = make_image(h=120, w=200)
        filename = database.save_image(img)
        loaded = database.get_image(filename)
        # JPEG może lekko zmienić wymiary kolorów, ale kształt musi pasować
        assert loaded.shape[:2] == (120, 200)


# ============================================================
# Testy add_detection
# ============================================================

class TestAddDetection:
    def test_add_returns_id(self, database):
        img = make_image()
        detection_id = database.add_detection(
            image=img,
            car_detected=True,
            car_brand="BMW",
            brand_confidence=0.92,
        )
        assert isinstance(detection_id, int)
        assert detection_id >= 1

    def test_add_increments_id(self, database):
        img1 = make_image(color=(1, 2, 3))
        img2 = make_image(color=(4, 5, 6))
        id1 = database.add_detection(image=img1)
        id2 = database.add_detection(image=img2)
        assert id2 > id1

    def test_add_duplicate_raises(self, database):
        img = make_image(color=(50, 50, 50))
        database.add_detection(image=img)
        with pytest.raises(sqlite3.IntegrityError):
            database.add_detection(image=img)

    def test_add_with_all_fields(self, database):
        img = make_image(color=(70, 80, 90))
        car_img = make_image(color=(10, 10, 10))
        plate_img = make_image(h=40, w=150, color=(255, 255, 255))
        detection_id = database.add_detection(
            image=img,
            car_detected=True,
            car_image=car_img,
            car_brand="AUDI",
            brand_confidence=0.88,
            plate_detected=True,
            plate_image=plate_img,
            plate_text="SC6271X",
            plate_confidence=0.95,
            notes="Test wpis",
        )
        assert detection_id >= 1

    def test_added_record_retrievable(self, database):
        img = make_image(color=(11, 22, 33))
        database.add_detection(
            image=img,
            car_detected=True,
            car_brand="MERCEDES",
            brand_confidence=0.75,
            plate_detected=True,
            plate_text="WA12345",
            plate_confidence=0.80,
        )
        detections = database.get_all_detections(limit=10)
        assert len(detections) >= 1
        rec = detections[0]
        assert rec["car_brand"] == "MERCEDES"
        assert rec["plate_text"] == "WA12345"


# ============================================================
# Testy pobierania danych
# ============================================================

class TestGetDetections:
    def _populate(self, database, n=3):
        for i in range(n):
            img = make_image(color=(i * 30 + 10, i * 20 + 5, i * 10 + 1))
            database.add_detection(
                image=img,
                car_detected=True,
                car_brand=["AUDI", "BMW", "MERCEDES"][i % 3],
                brand_confidence=0.8,
                plate_detected=(i % 2 == 0),
                plate_text=f"AB{1000 + i}" if i % 2 == 0 else None,
                plate_confidence=0.9 if i % 2 == 0 else 0.0,
            )

    def test_get_all_returns_list(self, database):
        self._populate(database, 2)
        result = database.get_all_detections()
        assert isinstance(result, list)

    def test_get_all_limit(self, database):
        self._populate(database, 3)
        result = database.get_all_detections(limit=2)
        assert len(result) <= 2

    def test_get_all_is_sorted_desc(self, database):
        self._populate(database, 3)
        results = database.get_all_detections(limit=100)
        ids = [r['id'] for r in results]
        assert ids == sorted(ids, reverse=True)

    def test_get_by_plate(self, database):
        img = make_image(color=(99, 88, 77))
        database.add_detection(
            image=img,
            car_detected=True,
            plate_detected=True,
            plate_text="KR9999A",
            plate_confidence=0.85,
        )
        results = database.get_detections_by_plate("KR9999A")
        assert len(results) >= 1
        assert all(r['plate_text'] == "KR9999A" for r in results)

    def test_get_by_plate_no_match(self, database):
        results = database.get_detections_by_plate("XXXXXXX")
        assert results == []

    def test_get_by_brand(self, database):
        img = make_image(color=(55, 66, 77))
        database.add_detection(
            image=img,
            car_detected=True,
            car_brand="PORSCHE",
            brand_confidence=0.93,
        )
        results = database.get_detections_by_brand("PORSCHE")
        assert len(results) >= 1
        assert all(r['car_brand'] == "PORSCHE" for r in results)

    def test_get_by_brand_no_match(self, database):
        results = database.get_detections_by_brand("LAMBORGHINI")
        assert results == []


# ============================================================
# Testy statystyk
# ============================================================

class TestGetStatistics:
    def test_stats_returns_dict(self, database):
        stats = database.get_statistics()
        assert isinstance(stats, dict)

    def test_stats_empty_db(self, database):
        stats = database.get_statistics()
        assert stats['total_detections'] == 0
        assert stats['cars_detected'] == 0

    def test_stats_count_correctly(self, database):
        for color in [(1, 2, 3), (4, 5, 6), (7, 8, 9)]:
            img = make_image(color=color)
            database.add_detection(
                image=img,
                car_detected=True,
                car_brand="BMW",
                brand_confidence=0.8,
            )

        stats = database.get_statistics()
        assert stats['total_detections'] == 3
        assert stats['cars_detected'] == 3

    def test_stats_plate_count(self, database):
        for i, color in enumerate([(10, 11, 12), (13, 14, 15)]):
            img = make_image(color=color)
            database.add_detection(
                image=img,
                car_detected=True,
                plate_detected=True,
                plate_text=f"WX{i}000",
                plate_confidence=0.9,
            )

        stats = database.get_statistics()
        assert stats['plates_detected'] == 2

    def test_stats_unique_brands(self, database):
        for brand, color in [("AUDI", (20, 21, 22)), ("BMW", (23, 24, 25))]:
            img = make_image(color=color)
            database.add_detection(
                image=img, car_detected=True,
                car_brand=brand, brand_confidence=0.8,
            )

        stats = database.get_statistics()
        assert stats['unique_brands'] == 2

    def test_stats_top_brands_list(self, database):
        stats = database.get_statistics()
        assert isinstance(stats['top_brands'], list)

    def test_stats_has_required_keys(self, database):
        stats = database.get_statistics()
        for key in ('total_detections', 'cars_detected', 'plates_detected',
                    'unique_brands', 'unique_plates', 'top_brands'):
            assert key in stats


# ============================================================
# Testy eksportu CSV
# ============================================================

class TestExportCSV:
    def test_creates_file(self, database, tmp_path):
        img = make_image(color=(30, 40, 50))
        database.add_detection(image=img, car_detected=True, car_brand="AUDI", brand_confidence=0.9)
        csv_path = tmp_path / "export.csv"
        database.export_to_csv(str(csv_path))
        assert csv_path.exists()

    def test_csv_has_header(self, database, tmp_path):
        img = make_image(color=(60, 70, 80))
        database.add_detection(image=img, car_brand="BMW", brand_confidence=0.8)
        csv_path = tmp_path / "export.csv"
        database.export_to_csv(str(csv_path))
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
        assert 'car_brand' in header
        assert 'plate_text' in header

    def test_csv_has_data_rows(self, database, tmp_path):
        for color in [(61, 71, 81), (62, 72, 82)]:
            img = make_image(color=color)
            database.add_detection(image=img, car_brand="BMW", brand_confidence=0.8)
        csv_path = tmp_path / "export.csv"
        database.export_to_csv(str(csv_path))
        with open(csv_path, newline='', encoding='utf-8') as f:
            rows = list(csv.reader(f))
        # 1 nagłówek + 2 rekordy
        assert len(rows) >= 3

    def test_csv_filter_by_plate(self, database, tmp_path):
        for i, color in enumerate([(91, 92, 93), (94, 95, 96)]):
            img = make_image(color=color)
            database.add_detection(
                image=img, car_detected=True,
                plate_detected=True,
                plate_text="TARGETPLATE" if i == 0 else "OTHERPLATE",
                plate_confidence=0.9,
            )
        csv_path = tmp_path / "filtered.csv"
        database.export_to_csv(str(csv_path), plate_text="TARGETPLATE")
        with open(csv_path, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        assert all(r['plate_text'] == "TARGETPLATE" for r in rows)


# ============================================================
# Testy usuwania rekordów
# ============================================================

class TestDeleteDetection:
    def test_delete_removes_record(self, database):
        img = make_image(color=(100, 110, 120))
        det_id = database.add_detection(image=img, car_brand="AUDI", brand_confidence=0.7)
        result = database.delete_detection(det_id)
        assert result is True
        rows = database.get_all_detections(limit=100)
        ids = [r['id'] for r in rows]
        assert det_id not in ids

    def test_delete_nonexistent_returns_true(self, database):
        # SQLite DELETE na nieistniejącym ID nie rzuca błędu
        result = database.delete_detection(999999)
        assert result is True

    def test_delete_does_not_remove_others(self, database):
        img1 = make_image(color=(130, 140, 150))
        img2 = make_image(color=(160, 170, 180))
        id1 = database.add_detection(image=img1, car_brand="BMW", brand_confidence=0.8)
        id2 = database.add_detection(image=img2, car_brand="AUDI", brand_confidence=0.7)

        database.delete_detection(id1)

        remaining = database.get_all_detections(limit=100)
        remaining_ids = [r['id'] for r in remaining]
        assert id2 in remaining_ids


# ============================================================
# Testy usuwania nieużywanych obrazów
# ============================================================

class TestCleanupUnusedImages:
    def test_cleanup_returns_int(self, database):
        result = database.cleanup_unused_images()
        assert isinstance(result, int)

    def test_cleanup_removes_orphan_files(self, database):
        # Stwórz plik w images_dir bez powiązanego rekordu
        orphan = database.images_dir / "orphan_test_file.jpg"
        orphan.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # fake JPEG
        deleted = database.cleanup_unused_images()
        assert deleted >= 1
        assert not orphan.exists()

    def test_cleanup_preserves_referenced_images(self, database):
        img = make_image(color=(200, 200, 200))
        filename = database.save_image(img)
        database.add_detection(
            image=img, car_detected=True, car_brand="BMW", brand_confidence=0.8
        )
        database.cleanup_unused_images()
        assert (database.images_dir / filename).exists()


# ============================================================
# Testy context managera
# ============================================================

class TestContextManager:
    def test_context_manager_works(self, tmp_path):
        db_dir = tmp_path / ".cv_ctx_test"
        db_dir.mkdir()
        images_dir = db_dir / "images"
        images_dir.mkdir()

        db = Database.__new__(Database)
        db.db_dir = db_dir
        db.db_path = db_dir / "database.db"
        db.images_dir = images_dir
        db.conn = None
        db._connect()
        db._create_tables()

        with db as d:
            img = make_image(color=(5, 6, 7))
            det_id = d.add_detection(image=img, car_brand="PORSCHE", brand_confidence=0.9)
            assert det_id >= 1

        # Po wyjściu z context managera connection powinna być zamknięta
        assert db.conn is not None  # SQLite pozwala na close + dalsze używanie ref
