"""
Moduł detekcji pojazdów przy użyciu YOLOv8.
Odpowiada za wykrywanie samochodów na obrazie i wybór najlepszego kandydata.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import Tuple, Optional, List

# Patch dla PyTorch 2.6+ - umożliwia ładowanie modeli YOLO
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


class CarDetector:
    """Klasa do wykrywania pojazdów na obrazie przy użyciu YOLOv8."""
    
    # Klasy COCO odpowiadające pojazdom
    VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    
    def __init__(self, model_path: str = 'yolov8s.pt'):
        """
        Inicjalizacja detektora pojazdów.
        
        Args:
            model_path: Ścieżka do modelu YOLOv8 (domyślnie yolov8s.pt)
        """
        self.model = YOLO(model_path)
    
    def detect_vehicles(self, image: np.ndarray) -> List[dict]:
        """
        Wykrywa wszystkie pojazdy na obrazie.
        
        Args:
            image: Obraz wejściowy (BGR)
            
        Returns:
            Lista słowników z danymi wykrytych pojazdów
        """
        results = self.model(image)[0]
        vehicles = []
        
        for detection in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in self.VEHICLE_CLASSES:
                bbox = [int(x1), int(y1), int(x2), int(y2)]
                area = (x2 - x1) * (y2 - y1)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                vehicles.append({
                    'bbox': bbox,
                    'score': score,
                    'class_id': int(class_id),
                    'area': area,
                    'center': (center_x, center_y)
                })
        
        return vehicles
    
    def select_best_vehicle(self, vehicles: List[dict], image_shape: Tuple[int, int]) -> Optional[dict]:
        """
        Wybiera najlepszy pojazd spośród wykrytych.
        Kryteria: największy bounding box, a jeśli podobne - najbardziej wycentrowany.
        
        Args:
            vehicles: Lista wykrytych pojazdów
            image_shape: Wymiary obrazu (height, width)
            
        Returns:
            Słownik z danymi najlepszego pojazdu lub None
        """
        if not vehicles:
            return None
        
        if len(vehicles) == 1:
            return vehicles[0]
        
        # Sortuj po powierzchni malejąco
        sorted_vehicles = sorted(vehicles, key=lambda v: v['area'], reverse=True)
        
        # Sprawdź czy największe są podobne rozmiarem (w granicach 20%)
        largest = sorted_vehicles[0]
        similar_size = [largest]
        
        for v in sorted_vehicles[1:]:
            if v['area'] >= largest['area'] * 0.8:
                similar_size.append(v)
            else:
                break
        
        if len(similar_size) == 1:
            return largest
        
        # Wybierz najbardziej wycentrowany
        img_height, img_width = image_shape[:2]
        img_center_x = img_width / 2
        img_center_y = img_height / 2
        
        best_vehicle = None
        min_distance = float('inf')
        
        for v in similar_size:
            cx, cy = v['center']
            distance = np.sqrt((cx - img_center_x) ** 2 + (cy - img_center_y) ** 2)
            if distance < min_distance:
                min_distance = distance
                best_vehicle = v
        
        return best_vehicle
    
    def detect_and_crop(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[dict], np.ndarray]:
        """
        Wykrywa pojazdy, wybiera najlepszy i wycina go z obrazu.
        
        Args:
            image: Obraz wejściowy (BGR)
            
        Returns:
            Tuple zawierająca:
            - Wycięty fragment z pojazdem (lub None)
            - Dane wykrytego pojazdu (lub None)
            - Obraz z zaznaczonym bounding boxem
        """
        vehicles = self.detect_vehicles(image)
        best_vehicle = self.select_best_vehicle(vehicles, image.shape)
        
        # Kopia obrazu do rysowania
        annotated_image = image.copy()
        
        if best_vehicle is None:
            return None, None, annotated_image
        
        # Rysuj bounding box
        x1, y1, x2, y2 = best_vehicle['bbox']
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Dodaj etykietę
        label = f"Samochod: {best_vehicle['score']:.2f}"
        cv2.putText(annotated_image, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Wytnij pojazd
        crop = image[y1:y2, x1:x2].copy()
        
        return crop, best_vehicle, annotated_image
