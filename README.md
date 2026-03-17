# REBUILD - Aplikacja do rozpoznawania pojazdów

Desktopowa aplikacja z graficznym interfejsem użytkownika do automatycznej analizy samochodów na zdjęciach.

## Co działa

### Detekcja pojazdów
- Wykrywanie samochodów na zdjęciu (YOLOv8s, klasy COCO)
- Automatyczny wybór najlepszego pojazdu (największy lub najbardziej wycentrowany)
- Rysowanie bounding boxa i wycięcie (crop) obszaru z pojazdem

### Klasyfikacja marki
- Rozpoznawanie marki pojazdu przy użyciu modelu MobileNetV2
- Obsługiwane marki: **AUDI, BMW, MERCEDES, PORSCHE, VOLKSWAGEN**
- Wyświetlanie pewności predykcji (w procentach)

### ANPR – rozpoznawanie tablic rejestracyjnych
- Wykrywanie tablicy na obrazie (własny model YOLOv8 – `anpr_best.pt`)
- Odczyt tekstu z tablicy przez PaddleOCR
- Obsługa tablic europejskich (polskie, niemieckie itp.)
- Wycięcie i wyświetlenie wykrytej tablicy

### Interfejs graficzny (GUI)
- Ciemny motyw (dark theme) oparty na PySide6
- Wczytywanie obrazu przez dialog wyboru pliku
- Wyświetlanie: oryginalnego obrazu z bbox, cropu auta, cropu tablicy
- Wyniki tekstowe: marka, pewność, numer tablicy
- Przetwarzanie w osobnym wątku (UI nie zawiesza się podczas analizy)

## Uruchomienie

```bash
cd car_vision_app
python main.py
```

### Opcjonalne parametry
```
--model, -m       Ścieżka do pliku z wagami MobileNetV2 (domyślnie: car_detector_model/model.pth)
--classes, -c     Ścieżka do pliku JSON z mapowaniem klas (domyślnie: car_detector_model/label_map.json)
--plate-model, -p Ścieżka do modelu wykrywania tablic YOLO (domyślnie: car_vision_app/anpr_best.pt)
```

## Wymagania

- Python 3.10+
- PySide6
- PyTorch + torchvision
- Ultralytics (YOLOv8)
- OpenCV
- PaddleOCR + PaddlePaddle

```bash
pip install -r requirements.txt
```

## Struktura projektu

```
car_vision_app/
├── main.py              # Punkt wejścia, parsowanie argumentów
├── gui.py               # Interfejs graficzny (PySide6, dark theme)
├── detection.py         # Detekcja pojazdów (YOLOv8s)
├── classification.py    # Klasyfikacja marki (MobileNetV2)
├── anpr.py              # ANPR: wykrywanie tablicy + OCR (PaddleOCR)
├── anpr_best.pt         # Model wykrywania tablic (YOLOv8)
└── yolov8s.pt           # Model detekcji ogólnej (YOLOv8s)

car_detector_model/
├── model.pth            # Wagi modelu MobileNetV2
└── label_map.json       # Mapowanie klas (AUDI, BMW, MERCEDES, PORSCHE, VOLKSWAGEN)
```

