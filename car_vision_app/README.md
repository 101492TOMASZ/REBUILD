# 🚗 Aplikacja do rozpoznawania pojazdów

Desktopowa aplikacja z graficznym interfejsem użytkownika (PySide6) do automatycznego rozpoznawania:
- **Marki pojazdu** (przy użyciu MobileNetV2)
- **Tablicy rejestracyjnej** (ANPR z OCR)

## 📋 Wymagania

- Python 3.10+
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- PySide6
- EasyOCR

## 🔧 Instalacja

1. **Sklonuj lub pobierz projekt**

2. **Utwórz wirtualne środowisko (zalecane)**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# lub
venv\Scripts\activate  # Windows
```

3. **Zainstaluj zależności**
```bash
pip install -r requirements.txt
```

## 📁 Struktura projektu

```
car_vision_app/
├── __init__.py          # Inicjalizacja pakietu
├── main.py              # Główny punkt wejścia
├── gui.py               # Interfejs graficzny (PySide6)
├── detection.py         # Moduł detekcji pojazdów (YOLOv8)
├── classification.py    # Moduł klasyfikacji marki (MobileNetV2)
├── anpr.py              # Moduł ANPR (wykrywanie tablic + OCR)
├── requirements.txt     # Zależności
└── README.md            # Dokumentacja

car_detector_model/
├── model.pth            # Wagi modelu MobileNetV2
└── label_map.json       # Mapowanie klas (marek)

Automatic-License-Plate-Recognition-using-YOLOv8-main/
└── license_plate_detector.pt  # Model wykrywania tablic
```

## 🚀 Uruchomienie

### Podstawowe uruchomienie
```bash
cd car_vision_app
python main.py
```

### Z własnymi ścieżkami do modeli
```bash
python main.py --model /ścieżka/do/model.pth --classes /ścieżka/do/classes.json
```

### Dostępne parametry
```
--model, -m       Ścieżka do pliku z wagami modelu MobileNetV2
--classes, -c     Ścieżka do pliku JSON z mapowaniem klas
--plate-model, -p Ścieżka do modelu wykrywania tablic YOLO
```

## 🖥️ Interfejs użytkownika

### Sekcja wejściowa
- **Przycisk "Wczytaj zdjęcie"** - otwiera dialog wyboru pliku obrazu
- **Podgląd** - wyświetla wczytany obraz

### Przetwarzanie
- **Przycisk "Analizuj"** - uruchamia oba pipeline'y

### Wizualizacja wyników
- Obraz z zaznaczonym bounding boxem wykrytego pojazdu
- Wycięty (crop) fragment z samochodem
- Wycięta tablica rejestracyjna (jeśli wykryta)

### Wyniki tekstowe
- Nazwa marki pojazdu
- Pewność predykcji (w procentach)
- Numer tablicy rejestracyjnej

## 🔄 Pipeline'y

### Pipeline 1 - Wykrywanie auta i rozpoznawanie marki
1. Wykrywanie pojazdów na obrazie (YOLOv8s)
2. Wybór najlepszego pojazdu (największy lub najbardziej wycentrowany)
3. Wycięcie (crop) wykrytego samochodu
4. Klasyfikacja marki (MobileNetV2)

### Pipeline 2 - Wykrywanie tablicy rejestracyjnej
1. Wykrywanie tablicy na obrazie samochodu (YOLO)
2. Wycięcie tablicy
3. Preprocessing (binaryzacja, skalowanie)
4. Odczyt znaków (EasyOCR)
5. Formatowanie tekstu tablicy

## 📝 Obsługiwane marki

Zależnie od wytrenowanego modelu. Domyślnie:
- AUDI
- BMW
- MERCEDES
- PORSCHE
- VOLKSWAGEN

## ⚠️ Uwagi

- Przy pierwszym uruchomieniu modele YOLOv8 zostaną automatycznie pobrane
- EasyOCR pobiera modele językowe przy pierwszym użyciu
- Dla lepszej wydajności zalecane jest GPU z obsługą CUDA

## 📄 Licencja

MIT License
