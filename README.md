# CarVision AI — Rozpoznawanie pojazdów

Desktopowa aplikacja z GUI (PySide6) do automatycznej analizy samochodów na zdjęciach: wykrywanie pojazdu, klasyfikacja marki i odczyt tablicy rejestracyjnej.

## Co działa

### 1. Detekcja pojazdów (`detection.py`)
- Wykrywanie samochodów, motocykli, autobusów i ciężarówek (YOLOv8s, klasy COCO 2/3/5/7)
- Automatyczny wybór najlepszego pojazdu — największy bounding box, a przy podobnych rozmiarach najbardziej wycentrowany
- Rysowanie bounding boxa na obrazie i wycięcie (crop) obszaru z pojazdem

### 2. Klasyfikacja marki (`classification.py`)
- Rozpoznawanie marki pojazdu — model MobileNetV2 z modyfikowaną głową klasyfikacyjną
- Obsługiwane marki: **AUDI, BMW, MERCEDES, PORSCHE, VOLKSWAGEN**
- Pewność predykcji (softmax) wyświetlana w procentach
- Top-K predykcji (domyślnie top 3)
- **Grad-CAM** — mapa ciepła pokazująca które obszary obrazu wpłynęły na decyzję klasyfikatora (ostatnia warstwa konwolucyjna MobileNetV2)

### 3. ANPR — rozpoznawanie tablic rejestracyjnych (`anpr.py`)
- Wykrywanie tablicy na wyciętym croppie pojazdu (własny model YOLOv8 — `anpr_best.pt`)
- Odczyt tekstu przez **PaddleOCR** (leniwa inicjalizacja, GPU auto-detekcja)
- Preprocessing wielowariantowy — 4 warianty obrazu tablicy (surowy, CLAHE+wyostrzenie, Otsu, adaptacyjny threshold)
- Korekcja typowych błędów OCR (0↔O, 1↔I, 8↔B, H↔W itp.)
- Walidacja formatu tablic europejskich (polskie, niemieckie, brytyjskie)
- Automatyczne usuwanie kodów krajów z niebieskiego pasa EU
- Formatowanie tekstu tablicy (np. WOB-AW642)

### 4. Baza danych (`database.py`)
- Lokalna baza SQLite w `~/.carvision/database.db`
- Przechowywanie wyników: marka, pewność, tablica, pewność OCR, obrazy (oryginał, crop, tablica, Grad-CAM)
- Obrazy zapisywane jako pliki JPG w `~/.carvision/images/` (deduplikacja po hashach MD5)
- Filtrowanie po marce, tablicy; paginacja wyników
- Statystyki (łączna liczba, rozkład marek, unikalne tablice)
- Eksport do CSV
- Oznaczanie błędnych wyników (`is_incorrect`)

### 5. Interfejs graficzny (`gui.py`)
- **Ciemny motyw** (dark theme) — paleta #111827 / #1f2937, zaokrąglone karty, gradienty na przyciskach
- Wczytywanie obrazu przez dialog lub **drag & drop**
- Asynchroniczne ładowanie modeli z paskiem postępu (nie blokuje UI)
- Równoległa analiza — po detekcji pojazdu jednocześnie: klasyfikacja marki (BrandWorker) + odczyt tablicy (ANPRWorker)
- Wyświetlanie wyników: marka z kolorową pewnością (zielony ≥80%, żółty ≥50%, czerwony <50%), stylizowana tablica rejestracyjna
- **Okno Crops** — niemodalny panel z 3 kafelkami: detekcja, wycięcie, tablica
- **Heatmapa (Grad-CAM)** — dialog z wizualizacją na co patrzył klasyfikator
- **Przetwarzanie wsadowe (Batch)** — analiza wielu zdjęć naraz z tabelą wyników i auto-zapisem do bazy
- **Historia** — dialog z tabelą wszystkich detekcji, filtry po marce/tablicy, eksport CSV
- Zapis do bazy jednym przyciskiem

### 6. Testy (`tests/`)
- **test_detection.py** — testy jednostkowe `CarDetector` (mockowany YOLO): stałe klas, `select_best_vehicle`, obsługa pustych list
- **test_classification.py** — testy `BrandClassifier`: ładowanie klas, `preprocess` (shape, dtype, batch), odwrotne mapowanie
- **test_anpr.py** — testy algorytmiczne `ANPRModule`: `clean_plate_text`, `validate_plate_format` (PL/DE/UK/nieprawidłowe)
- **test_database.py** — testy `Database`: inicjalizacja schematu, indeksy, hash, zapis/odczyt obrazów
- **test_batch_worker.py** — testy `BatchWorker` z mockami detektora, klasyfikatora, ANPR i bazy
- Testy integracyjne oznaczone `@pytest.mark.integration` (wymagają prawdziwych modeli)

### 7. Narzędzia pomocnicze (`tests/`)
- `bing_scraper.py`, `download_car_images.py` — pobieranie zdjęć samochodów
- `download_plate_images.py` — pobieranie zdjęć tablic
- `downloader_german.py` — scraper zdjęć niemieckich marek
- `split_and_dedupe.py` — podział i deduplikacja datasetu

## Uruchomienie

```bash
cd car_vision_app
python main.py
```

### Opcjonalne parametry
```
--model, -m       Ścieżka do wag MobileNetV2        (domyślnie: car_detector_model/model.pth)
--classes, -c     Ścieżka do label_map.json          (domyślnie: car_detector_model/label_map.json)
--plate-model, -p Ścieżka do modelu YOLO tablic      (domyślnie: car_vision_app/anpr_best.pt)
```

## Wymagania

- Python 3.10+
- PySide6 ≥ 6.5
- PyTorch ≥ 2.0 + torchvision ≥ 0.15
- Ultralytics ≥ 8.0 (YOLOv8)
- OpenCV ≥ 4.8
- PaddlePaddle 2.6.2 + PaddleOCR 2.9.1

```bash
pip install -r requirements.txt
```

## Struktura projektu

```
car_vision_app/
├── __init__.py          # Pakiet, leniwy import modułów
├── main.py              # Punkt wejścia, parsowanie argumentów CLI
├── gui.py               # Interfejs graficzny (PySide6, dark theme, wątki, batch, historia)
├── detection.py         # Detekcja pojazdów (YOLOv8s, COCO)
├── classification.py    # Klasyfikacja marki (MobileNetV2) + Grad-CAM
├── anpr.py              # ANPR: detekcja tablicy (YOLOv8) + OCR (PaddleOCR)
├── database.py          # Baza danych SQLite + przechowywanie obrazów
├── anpr_best.pt         # Model wykrywania tablic (YOLOv8, fine-tuned)
└── yolov8s.pt           # Model detekcji ogólnej (YOLOv8s, COCO)

car_detector_model/
├── model.pth            # Wagi MobileNetV2 (5 klas marek)
└── label_map.json       # Mapowanie klas: AUDI=0, BMW=1, MERCEDES=2, PORSCHE=3, VOLKSWAGEN=4

dataset_german/          # Dane treningowe (AUDI, BMW, MERCEDES, PORSCHE, VOLKSWAGEN)

tests/
├── conftest.py          # Wspólne fixtures (obrazy syntetyczne, ścieżki modeli)
├── test_detection.py    # Testy detekcji pojazdów
├── test_classification.py # Testy klasyfikacji marki
├── test_anpr.py         # Testy ANPR (czyszczenie, walidacja, korekcja OCR)
├── test_database.py     # Testy bazy danych
├── test_batch_worker.py # Testy przetwarzania wsadowego
└── images/              # Zdjęcia testowe (5 marek × 5 perspektyw)
```

## Pipeline analizy

1. **Detekcja** — YOLOv8s wykrywa pojazdy → wybór najlepszego → crop
2. **Klasyfikacja** *(równolegle)* — MobileNetV2 rozpoznaje markę na croppie
3. **ANPR** *(równolegle)* — YOLOv8 wykrywa tablicę → preprocessing → PaddleOCR → korekcja → walidacja
4. **Wyświetlenie** — wyniki w GUI + opcjonalnie zapis do bazy

## Testy

```bash
python -m pytest tests/ -v
python -m pytest tests/ -v -m integration   # wymaga prawdziwych modeli
```

