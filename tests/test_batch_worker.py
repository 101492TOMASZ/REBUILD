"""
Testy jednostkowe BatchWorker i BatchProgressDialog (z gui.py).

BatchWorker jest testowany bez GUI (QApplication nie jest wymagana dla logiki).
Sygnały Qt są mockowane aby testy mogły działać bez event loop.
"""

import sys
import os
import pytest
import numpy as np
import cv2
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import requires_models


# ============================================================
# Import BatchWorker bez wywołania QApplication
# ============================================================

# Patchujemy PySide6 zanim zostanie zaimportowany przez gui.py
# jeśli środowisko nie ma display (CI/headless)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def batch_worker_class():
    """Importuje BatchWorker opóźniając import PySide6."""
    from car_vision_app.gui import BatchWorker
    return BatchWorker


# ============================================================
# Fixture — mock detektora, klasyfikatora, ANPR i bazy
# ============================================================

@pytest.fixture
def mock_detector():
    d = MagicMock()
    # Zwraca (crop, vehicle_data, annotated)
    crop = np.ones((100, 200, 3), dtype=np.uint8) * 128
    vehicle_data = {'bbox': [10, 10, 210, 110], 'score': 0.9}
    annotated = np.ones((300, 400, 3), dtype=np.uint8) * 100
    d.detect_and_crop.return_value = (crop, vehicle_data, annotated)
    return d


@pytest.fixture
def mock_detector_no_car():
    d = MagicMock()
    annotated = np.ones((300, 400, 3), dtype=np.uint8) * 50
    d.detect_and_crop.return_value = (None, None, annotated)
    return d


@pytest.fixture
def mock_classifier():
    c = MagicMock()
    c.predict.return_value = ("BMW", 88.5)
    return c


@pytest.fixture
def mock_anpr():
    a = MagicMock()
    a.process.return_value = {
        'detected': True,
        'text': 'SC6271X',
        'confidence': 0.97,
        'plate_crop': np.ones((40, 150, 3), dtype=np.uint8) * 255,
        'bbox': [5, 5, 155, 45],
    }
    return a


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.add_detection.return_value = 42
    return db


@pytest.fixture
def image_files(tmp_path):
    """Tworzy kilka tymczasowych plików obrazów do testowania."""
    files = []
    for i in range(3):
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        img[:] = (i * 50 + 30, i * 40 + 20, i * 30 + 10)
        p = tmp_path / f"test_car_{i}.jpg"
        cv2.imwrite(str(p), img)
        files.append(str(p))
    return files


@pytest.fixture
def nonexistent_file(tmp_path):
    return str(tmp_path / "does_not_exist.jpg")


# ============================================================
# Testy inicjalizacji BatchWorker
# ============================================================

class TestBatchWorkerInit:
    def test_file_paths_stored(self, batch_worker_class, image_files,
                               mock_detector, mock_classifier, mock_anpr, mock_db):
        worker = batch_worker_class(
            image_files, mock_detector, mock_classifier, mock_anpr, mock_db
        )
        assert worker.file_paths == image_files

    def test_not_cancelled_on_init(self, batch_worker_class, image_files,
                                   mock_detector, mock_classifier, mock_anpr, mock_db):
        worker = batch_worker_class(
            image_files, mock_detector, mock_classifier, mock_anpr, mock_db
        )
        assert worker._cancelled is False

    def test_cancel_sets_flag(self, batch_worker_class, image_files,
                              mock_detector, mock_classifier, mock_anpr, mock_db):
        worker = batch_worker_class(
            image_files, mock_detector, mock_classifier, mock_anpr, mock_db
        )
        worker.cancel()
        assert worker._cancelled is True


# ============================================================
# Testy run() — bezpośrednie wywołanie zamiast wątku
# ============================================================

class TestBatchWorkerRun:
    """Testy wywołujące worker.run() bezpośrednio (bez start())."""

    def _run_worker(self, batch_worker_class, files,
                    detector, classifier, anpr, db):
        """Uruchamia worker z zamockowanymi sygnałami i zwraca wyniki."""
        worker = batch_worker_class(files, detector, classifier, anpr, db)

        collected_results = []
        started_indices = []
        done_indices = []
        error_indices = []

        worker.image_started = MagicMock()
        worker.image_done = MagicMock()
        worker.batch_error = MagicMock()
        worker.all_done = MagicMock()

        worker.image_started.emit = lambda idx, path: started_indices.append(idx)
        worker.image_done.emit = lambda idx, result: done_indices.append((idx, result))
        worker.batch_error.emit = lambda idx, err: error_indices.append((idx, err))
        worker.all_done.emit = lambda results: collected_results.extend(results)

        worker.run()
        return collected_results, started_indices, done_indices, error_indices

    def test_all_done_called_with_results(self, batch_worker_class, image_files,
                                          mock_detector, mock_classifier, mock_anpr, mock_db):
        results, _, _, _ = self._run_worker(
            batch_worker_class, image_files,
            mock_detector, mock_classifier, mock_anpr, mock_db
        )
        assert len(results) == len(image_files)

    def test_each_image_started(self, batch_worker_class, image_files,
                                mock_detector, mock_classifier, mock_anpr, mock_db):
        _, started, _, _ = self._run_worker(
            batch_worker_class, image_files,
            mock_detector, mock_classifier, mock_anpr, mock_db
        )
        assert len(started) == len(image_files)

    def test_results_contain_path(self, batch_worker_class, image_files,
                                  mock_detector, mock_classifier, mock_anpr, mock_db):
        results, _, _, _ = self._run_worker(
            batch_worker_class, image_files,
            mock_detector, mock_classifier, mock_anpr, mock_db
        )
        for r in results:
            assert 'path' in r

    def test_car_detected_in_result(self, batch_worker_class, image_files,
                                    mock_detector, mock_classifier, mock_anpr, mock_db):
        results, _, _, _ = self._run_worker(
            batch_worker_class, image_files,
            mock_detector, mock_classifier, mock_anpr, mock_db
        )
        for r in results:
            assert 'car_detected' in r
            assert r['car_detected'] is True

    def test_brand_from_classifier(self, batch_worker_class, image_files,
                                   mock_detector, mock_classifier, mock_anpr, mock_db):
        results, _, _, _ = self._run_worker(
            batch_worker_class, image_files,
            mock_detector, mock_classifier, mock_anpr, mock_db
        )
        for r in results:
            assert r.get('brand') == "BMW"

    def test_plate_text_from_anpr(self, batch_worker_class, image_files,
                                  mock_detector, mock_classifier, mock_anpr, mock_db):
        results, _, _, _ = self._run_worker(
            batch_worker_class, image_files,
            mock_detector, mock_classifier, mock_anpr, mock_db
        )
        for r in results:
            assert r.get('plate_text') == "SC6271X"

    def test_saved_true_when_db_available(self, batch_worker_class, image_files,
                                         mock_detector, mock_classifier, mock_anpr, mock_db):
        results, _, _, _ = self._run_worker(
            batch_worker_class, image_files,
            mock_detector, mock_classifier, mock_anpr, mock_db
        )
        assert all(r.get('saved') is True for r in results)

    def test_saved_false_when_no_db(self, batch_worker_class, image_files,
                                    mock_detector, mock_classifier, mock_anpr):
        results, _, _, _ = self._run_worker(
            batch_worker_class, image_files,
            mock_detector, mock_classifier, mock_anpr, db=None
        )
        assert all(r.get('saved') is False for r in results)

    def test_no_car_detected_skips_classifier(self, batch_worker_class, image_files,
                                              mock_detector_no_car, mock_classifier,
                                              mock_anpr, mock_db):
        results, _, _, _ = self._run_worker(
            batch_worker_class, image_files,
            mock_detector_no_car, mock_classifier, mock_anpr, mock_db
        )
        # Klasyfikator nie powinien być wywoływany
        mock_classifier.predict.assert_not_called()
        for r in results:
            assert r['car_detected'] is False
            assert r.get('brand') is None

    def test_unreadable_file_gets_error_key(self, batch_worker_class, nonexistent_file,
                                            mock_detector, mock_classifier,
                                            mock_anpr, mock_db):
        results, _, _, _ = self._run_worker(
            batch_worker_class, [nonexistent_file],
            mock_detector, mock_classifier, mock_anpr, mock_db
        )
        assert len(results) == 1
        assert 'error' in results[0]

    def test_cancel_stops_processing(self, batch_worker_class, image_files,
                                     mock_detector, mock_classifier, mock_anpr, mock_db):
        worker = batch_worker_class(
            image_files, mock_detector, mock_classifier, mock_anpr, mock_db
        )
        # Anuluj przed uruchomieniem
        worker.cancel()

        collected = []
        worker.image_started = MagicMock()
        worker.image_done = MagicMock()
        worker.batch_error = MagicMock()
        worker.all_done = MagicMock()
        worker.all_done.emit = lambda r: collected.extend(r)
        worker.image_started.emit = lambda *a: None
        worker.image_done.emit = lambda *a: None
        worker.batch_error.emit = lambda *a: None

        worker.run()
        # Po cancel() żadne pliki nie powinny być przetworzone
        assert len(collected) == 0

    def test_db_save_called_for_each_image(self, batch_worker_class, image_files,
                                           mock_detector, mock_classifier,
                                           mock_anpr, mock_db):
        self._run_worker(
            batch_worker_class, image_files,
            mock_detector, mock_classifier, mock_anpr, mock_db
        )
        assert mock_db.add_detection.call_count == len(image_files)

    def test_db_exception_marks_not_saved(self, batch_worker_class, image_files,
                                          mock_detector, mock_classifier, mock_anpr):
        bad_db = MagicMock()
        bad_db.add_detection.side_effect = Exception("DB error")

        results, _, _, _ = self._run_worker(
            batch_worker_class, image_files,
            mock_detector, mock_classifier, mock_anpr, bad_db
        )
        assert all(r.get('saved') is False for r in results)

    def test_empty_file_list(self, batch_worker_class, mock_detector,
                             mock_classifier, mock_anpr, mock_db):
        results, started, done, errors = self._run_worker(
            batch_worker_class, [],
            mock_detector, mock_classifier, mock_anpr, mock_db
        )
        assert results == []
        assert started == []


# ============================================================
# Testy wyników — struktura słownika
# ============================================================

class TestBatchResultStructure:
    EXPECTED_KEYS_SUCCESS = {
        'path', 'car_detected', 'brand', 'brand_confidence',
        'plate_detected', 'plate_text', 'plate_confidence',
        'detection_id', 'saved',
    }

    def test_success_result_has_all_keys(self, batch_worker_class, image_files,
                                         mock_detector, mock_classifier,
                                         mock_anpr, mock_db):
        from car_vision_app.gui import BatchWorker
        worker = BatchWorker(
            image_files, mock_detector, mock_classifier, mock_anpr, mock_db
        )
        collected = []
        worker.image_started = MagicMock()
        worker.image_done = MagicMock()
        worker.batch_error = MagicMock()
        worker.all_done = MagicMock()
        worker.all_done.emit = lambda r: collected.extend(r)
        worker.image_started.emit = lambda *a: None
        worker.image_done.emit = lambda *a: None
        worker.batch_error.emit = lambda *a: None
        worker.run()

        for r in collected:
            if 'error' not in r:
                assert self.EXPECTED_KEYS_SUCCESS.issubset(r.keys()), \
                    f"Brakujące klucze: {self.EXPECTED_KEYS_SUCCESS - r.keys()}"

    def test_error_result_has_path_and_error(self, batch_worker_class,
                                             nonexistent_file, mock_detector,
                                             mock_classifier, mock_anpr, mock_db):
        from car_vision_app.gui import BatchWorker
        worker = BatchWorker(
            [nonexistent_file], mock_detector, mock_classifier, mock_anpr, mock_db
        )
        collected = []
        worker.image_started = MagicMock()
        worker.image_done = MagicMock()
        worker.batch_error = MagicMock()
        worker.all_done = MagicMock()
        worker.all_done.emit = lambda r: collected.extend(r)
        worker.image_started.emit = lambda *a: None
        worker.image_done.emit = lambda *a: None
        worker.batch_error.emit = lambda *a: None
        worker.run()

        assert len(collected) == 1
        assert 'path' in collected[0]
        assert 'error' in collected[0]
