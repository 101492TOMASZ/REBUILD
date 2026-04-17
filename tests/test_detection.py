"""
Testy jednostkowe modułu detekcji pojazdów (CarDetector).

Testy jednostkowe nie ładują modelu YOLO — mockują go.
Testy integracyjne (oznaczone @pytest.mark.integration) wymagają yolov8s.pt.
"""

import sys
import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))
from car_vision_app.detection import CarDetector
from conftest import requires_models


# ============================================================
# Fixture — CarDetector bez załadowanego modelu
# ============================================================

@pytest.fixture
def detector_bare():
    """CarDetector z zamockowanym modelem YOLO."""
    with patch("car_vision_app.detection.YOLO") as mock_yolo:
        mock_yolo.return_value = MagicMock()
        d = CarDetector(model_path="fake_model.pt")
    return d


# ============================================================
# Testy stałych klasy
# ============================================================

class TestCarDetectorConstants:
    def test_vehicle_classes_contains_car(self, detector_bare):
        assert 2 in detector_bare.VEHICLE_CLASSES  # car

    def test_vehicle_classes_contains_bus(self, detector_bare):
        assert 5 in detector_bare.VEHICLE_CLASSES  # bus

    def test_vehicle_classes_contains_truck(self, detector_bare):
        assert 7 in detector_bare.VEHICLE_CLASSES  # truck

    def test_vehicle_classes_does_not_contain_person(self, detector_bare):
        assert 0 not in detector_bare.VEHICLE_CLASSES  # person


# ============================================================
# Testy select_best_vehicle
# ============================================================

class TestSelectBestVehicle:
    IMAGE_SHAPE = (480, 640, 3)

    def _make_vehicle(self, x1, y1, x2, y2, score=0.9, class_id=2):
        area = (x2 - x1) * (y2 - y1)
        return {
            'bbox': [x1, y1, x2, y2],
            'score': score,
            'class_id': class_id,
            'area': area,
            'center': ((x1 + x2) / 2, (y1 + y2) / 2),
        }

    def test_empty_list_returns_none(self, detector_bare):
        result = detector_bare.select_best_vehicle([], self.IMAGE_SHAPE)
        assert result is None

    def test_single_vehicle_returned(self, detector_bare):
        v = self._make_vehicle(100, 100, 400, 350)
        result = detector_bare.select_best_vehicle([v], self.IMAGE_SHAPE)
        assert result is v

    def test_largest_vehicle_preferred(self, detector_bare):
        small = self._make_vehicle(10, 10, 50, 50)     # area = 40*40 = 1600
        large = self._make_vehicle(50, 50, 500, 400)   # area = 450*350 = 157 500
        result = detector_bare.select_best_vehicle([small, large], self.IMAGE_SHAPE)
        assert result is large

    def test_centered_preferred_when_similar_size(self, detector_bare):
        # Obraz 640x480, środek = (320, 240)
        # Dwa pojazdy podobnego rozmiaru — wybierz bliższy centru
        center_v = self._make_vehicle(120, 90, 520, 390)   # center=(320,240)
        edge_v   = self._make_vehicle(0,   0,  400, 310)   # center=(200,155)
        # Upewnij się, że są podobnego rozmiaru (edge >= 80% center)
        assert edge_v['area'] >= center_v['area'] * 0.8
        result = detector_bare.select_best_vehicle([edge_v, center_v], self.IMAGE_SHAPE)
        assert result is center_v

    def test_multiple_vehicles_returns_one(self, detector_bare):
        vehicles = [
            self._make_vehicle(0,   0,  100, 100),
            self._make_vehicle(200, 100, 500, 380),
            self._make_vehicle(50,  50,  150, 150),
        ]
        result = detector_bare.select_best_vehicle(vehicles, self.IMAGE_SHAPE)
        assert result is not None
        assert result in vehicles


# ============================================================
# Testy detect_vehicles — mockowanie YOLO
# ============================================================

class TestDetectVehicles:
    def _make_yolo_detection(self, x1, y1, x2, y2, score, class_id):
        return [x1, y1, x2, y2, score, class_id]

    def test_returns_empty_on_no_detections(self, detector_bare):
        result_mock = MagicMock()
        result_mock.boxes.data.tolist.return_value = []
        detector_bare.model.return_value = [result_mock]

        img = np.zeros((300, 400, 3), dtype=np.uint8)
        detections = detector_bare.detect_vehicles(img)
        assert detections == []

    def test_ignores_non_vehicle_classes(self, detector_bare):
        # class_id=0 to osoba — powinna zostać pominięta
        result_mock = MagicMock()
        result_mock.boxes.data.tolist.return_value = [
            [10.0, 20.0, 100.0, 150.0, 0.95, 0.0],  # person
        ]
        detector_bare.model.return_value = [result_mock]

        img = np.zeros((300, 400, 3), dtype=np.uint8)
        detections = detector_bare.detect_vehicles(img)
        assert detections == []

    def test_detects_car_class(self, detector_bare):
        result_mock = MagicMock()
        result_mock.boxes.data.tolist.return_value = [
            [50.0, 60.0, 400.0, 350.0, 0.92, 2.0],  # car
        ]
        detector_bare.model.return_value = [result_mock]

        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        detections = detector_bare.detect_vehicles(img)
        assert len(detections) == 1
        assert detections[0]['class_id'] == 2
        assert abs(detections[0]['score'] - 0.92) < 0.001

    def test_detects_multiple_vehicles(self, detector_bare):
        result_mock = MagicMock()
        result_mock.boxes.data.tolist.return_value = [
            [10.0, 10.0, 200.0, 200.0, 0.88, 2.0],   # car
            [300.0, 50.0, 580.0, 280.0, 0.75, 7.0],  # truck
        ]
        detector_bare.model.return_value = [result_mock]

        img = np.ones((480, 640, 3), dtype=np.uint8) * 100
        detections = detector_bare.detect_vehicles(img)
        assert len(detections) == 2

    def test_detection_has_required_keys(self, detector_bare):
        result_mock = MagicMock()
        result_mock.boxes.data.tolist.return_value = [
            [10.0, 20.0, 300.0, 250.0, 0.90, 2.0],
        ]
        detector_bare.model.return_value = [result_mock]

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector_bare.detect_vehicles(img)
        assert len(detections) == 1
        d = detections[0]
        assert 'bbox' in d
        assert 'score' in d
        assert 'class_id' in d
        assert 'area' in d
        assert 'center' in d

    def test_bbox_values_are_integers(self, detector_bare):
        result_mock = MagicMock()
        result_mock.boxes.data.tolist.return_value = [
            [10.7, 20.3, 300.9, 250.1, 0.90, 2.0],
        ]
        detector_bare.model.return_value = [result_mock]

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector_bare.detect_vehicles(img)
        bbox = detections[0]['bbox']
        assert all(isinstance(v, int) for v in bbox)


# ============================================================
# Testy detect_and_crop
# ============================================================

class TestDetectAndCrop:
    def test_returns_none_crop_when_no_vehicle(self, detector_bare):
        result_mock = MagicMock()
        result_mock.boxes.data.tolist.return_value = []
        detector_bare.model.return_value = [result_mock]

        img = np.zeros((300, 400, 3), dtype=np.uint8)
        crop, vehicle_data, annotated = detector_bare.detect_and_crop(img)

        assert crop is None
        assert vehicle_data is None
        assert annotated is not None
        assert annotated.shape == img.shape

    def test_returns_crop_when_vehicle_found(self, detector_bare):
        result_mock = MagicMock()
        result_mock.boxes.data.tolist.return_value = [
            [50.0, 60.0, 400.0, 350.0, 0.92, 2.0],
        ]
        detector_bare.model.return_value = [result_mock]

        img = np.ones((480, 640, 3), dtype=np.uint8) * 200
        crop, vehicle_data, annotated = detector_bare.detect_and_crop(img)

        assert crop is not None
        assert vehicle_data is not None
        assert crop.shape[0] == 350 - 60
        assert crop.shape[1] == 400 - 50

    def test_annotated_image_same_shape(self, detector_bare):
        result_mock = MagicMock()
        result_mock.boxes.data.tolist.return_value = [
            [10.0, 10.0, 200.0, 180.0, 0.91, 2.0],
        ]
        detector_bare.model.return_value = [result_mock]

        img = np.ones((480, 640, 3), dtype=np.uint8) * 150
        _, _, annotated = detector_bare.detect_and_crop(img)

        assert annotated.shape == img.shape

    def test_original_image_not_modified(self, detector_bare):
        result_mock = MagicMock()
        result_mock.boxes.data.tolist.return_value = [
            [10.0, 10.0, 200.0, 180.0, 0.91, 2.0],
        ]
        detector_bare.model.return_value = [result_mock]

        img = np.ones((480, 640, 3), dtype=np.uint8) * 150
        original_copy = img.copy()
        detector_bare.detect_and_crop(img)

        np.testing.assert_array_equal(img, original_copy)


# ============================================================
# Testy integracyjne (wymagają yolov8s.pt)
# ============================================================

@requires_models
@pytest.mark.integration
class TestCarDetectorIntegration:
    """Testy integracyjne ładujące rzeczywisty model YOLOv8."""

    @pytest.fixture
    def detector(self):
        return CarDetector(model_path="yolov8s.pt")

    def test_model_loads(self, detector):
        assert detector.model is not None

    def test_detect_on_black_image_returns_empty(self, detector):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect_vehicles(img)
        assert isinstance(detections, list)

    def test_detect_and_crop_on_blank_returns_none(self, detector):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        crop, data, annotated = detector.detect_and_crop(img)
        assert crop is None
        assert data is None
