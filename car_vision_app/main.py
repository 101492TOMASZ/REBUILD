#!/usr/bin/env python3
"""
Aplikacja do rozpoznawania pojazdów - marka i tablica rejestracyjna.

Uruchomienie:
    python main.py
    
    lub z własnymi ścieżkami:
    python main.py --model /ścieżka/do/model.pth --classes /ścieżka/do/classes.json
"""

import argparse
import os
import sys
from pathlib import Path
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def main():
    parser = argparse.ArgumentParser(
        description='Aplikacja do rozpoznawania marki pojazdu i tablicy rejestracyjnej',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:
  python main.py
  python main.py --model ./model.pth --classes ./classes.json
  python main.py --plate-model ./license_plate_detector.pt
        """
    )
    
    # Domyślne ścieżki względem katalogu projektu
    base_dir = Path(__file__).parent.parent
    default_model = base_dir / 'car_detector_model' / 'model.pth'
    default_classes = base_dir / 'car_detector_model' / 'label_map.json'
    # Priorytet: anpr_best.pt w katalogu aplikacji, potem stary model
    app_dir = Path(__file__).parent
    _new_model = app_dir / 'anpr_best.pt'
    _old_model = base_dir / 'Automatic-License-Plate-Recognition-using-YOLOv8-main' / 'license_plate_detector.pt'
    default_plate_model = _new_model if _new_model.exists() else _old_model
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=str(default_model),
        help=f'Ścieżka do pliku z wagami modelu MobileNetV2 (domyślnie: {default_model})'
    )
    
    parser.add_argument(
        '--classes', '-c',
        type=str,
        default=str(default_classes),
        help=f'Ścieżka do pliku JSON z mapowaniem klas (domyślnie: {default_classes})'
    )
    
    parser.add_argument(
        '--plate-model', '-p',
        type=str,
        default=str(default_plate_model),
        help=f'Ścieżka do modelu wykrywania tablic YOLO (domyślnie: {default_plate_model})'
    )
    
    args = parser.parse_args()
    
    # Sprawdź czy pliki istnieją
    if not os.path.exists(args.model):
        print(f"❌ Błąd: Nie znaleziono pliku modelu: {args.model}")
        print("   Podaj prawidłową ścieżkę za pomocą parametru --model")
        sys.exit(1)
    
    if not os.path.exists(args.classes):
        print(f"❌ Błąd: Nie znaleziono pliku klas: {args.classes}")
        print("   Podaj prawidłową ścieżkę za pomocą parametru --classes")
        sys.exit(1)
    
    plate_model = args.plate_model if os.path.exists(args.plate_model) else None
    if plate_model is None:
        print(f"⚠️ Ostrzeżenie: Nie znaleziono modelu wykrywania tablic: {args.plate_model}")
        print("   Moduł ANPR będzie szukał modelu w domyślnej lokalizacji.")
    
    print("=" * 60)
    print("🚗 Aplikacja do rozpoznawania pojazdów")
    print("=" * 60)
    print(f"📁 Model klasyfikatora: {args.model}")
    print(f"📁 Plik klas: {args.classes}")
    print(f"📁 Model tablicy: {plate_model or 'domyślna lokalizacja'}")
    print("=" * 60)
    print("🚀 Uruchamianie aplikacji...")
    print()
    
    # Import i uruchomienie GUI
    try:
        from .gui import run_app
    except ImportError:
        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        from gui import run_app
    run_app(args.model, args.classes, plate_model)


if __name__ == "__main__":
    main()
