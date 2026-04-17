#!/usr/bin/env python3
"""
Scraper samochodów marek — Bing Images, bez klucza API.

Pobiera 200 zdjęć dla każdej marki (AUDI, BMW, MERCEDES, PORSCHE, VOLKSWAGEN),
podzielonych na widoki:
    front      — przód pojazdu
    rear       — tył pojazdu
    side       — bok pojazdu
    3quarter   — widok 3/4 (przód-bok)
    interior   — wnętrze / kokpit

Struktura wyjściowa:
    tests/images/cars/
        audi/
            front/        (~40 zdjęć)
            rear/         (~40 zdjęć)
            side/         (~40 zdjęć)
            3quarter/     (~40 zdjęć)
            interior/     (~40 zdjęć)
        bmw/
            ...

Użycie:
    python tests/download_car_images.py
    python tests/download_car_images.py --output tests/images/cars --count 200
    python tests/download_car_images.py --brands AUDI BMW --count 100
    python tests/download_car_images.py --views front rear --count 80
    python tests/download_car_images.py --workers 6
"""

import os
import sys
import argparse
import hashlib
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

# Upewnij się że katalog tests/ jest w sys.path (przy uruchomieniu z root)
sys.path.insert(0, str(Path(__file__).parent))
from bing_scraper import bing_search_urls, download_file  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# Definicja marek + widoków
# ============================================================

BRANDS = ["audi", "bmw", "mercedes", "porsche", "volkswagen"]

# Dla każdego widoku — lista zapytań w kolejności priorytetu.
# Liczba zapytań > 1 → dodatkowe wyniki gdy pierwsze się skończy.
VIEW_QUERIES: Dict[str, Dict[str, List[str]]] = {
    "audi": {
        "front": [
            "Audi car front view photo",
            "Audi sedan front exterior",
            "Audi A4 front photo",
            "Audi Q5 front view",
        ],
        "rear": [
            "Audi car rear view photo",
            "Audi back exterior photo",
            "Audi A6 rear view",
            "Audi Q7 back photo",
        ],
        "side": [
            "Audi car side view photo",
            "Audi lateral exterior profile",
            "Audi A3 side view",
            "Audi TT side photo",
        ],
        "3quarter": [
            "Audi car 3/4 front view photo",
            "Audi three quarter angle exterior",
            "Audi Q5 three quarter",
            "Audi front quarter view",
        ],
        "interior": [
            "Audi car interior cockpit photo",
            "Audi dashboard steering wheel",
            "Audi cabin interior shot",
            "Audi A4 interior photo",
        ],
    },
    "bmw": {
        "front": [
            "BMW car front view photo",
            "BMW sedan front exterior",
            "BMW 3 Series front photo",
            "BMW X5 front view",
        ],
        "rear": [
            "BMW car rear view photo",
            "BMW back exterior photo",
            "BMW 5 Series rear view",
            "BMW X3 back photo",
        ],
        "side": [
            "BMW car side view photo",
            "BMW lateral exterior profile",
            "BMW M3 side view",
            "BMW 7 Series side photo",
        ],
        "3quarter": [
            "BMW car 3/4 front view photo",
            "BMW three quarter angle exterior",
            "BMW X5 three quarter",
            "BMW front quarter view",
        ],
        "interior": [
            "BMW car interior cockpit photo",
            "BMW dashboard steering wheel",
            "BMW cabin interior shot",
            "BMW 3 Series interior photo",
        ],
    },
    "mercedes": {
        "front": [
            "Mercedes-Benz car front view photo",
            "Mercedes sedan front exterior",
            "Mercedes C-Class front photo",
            "Mercedes GLE front view",
        ],
        "rear": [
            "Mercedes-Benz car rear view photo",
            "Mercedes back exterior photo",
            "Mercedes E-Class rear view",
            "Mercedes AMG back photo",
        ],
        "side": [
            "Mercedes-Benz car side view photo",
            "Mercedes lateral exterior profile",
            "Mercedes S-Class side view",
            "Mercedes CLA side photo",
        ],
        "3quarter": [
            "Mercedes-Benz 3/4 front view photo",
            "Mercedes three quarter angle exterior",
            "Mercedes GLC three quarter view",
            "Mercedes front quarter photo",
        ],
        "interior": [
            "Mercedes-Benz interior cockpit photo",
            "Mercedes dashboard steering wheel",
            "Mercedes cabin interior shot",
            "Mercedes C-Class interior photo",
        ],
    },
    "porsche": {
        "front": [
            "Porsche car front view photo",
            "Porsche 911 front exterior",
            "Porsche Cayenne front photo",
            "Porsche Macan front view",
        ],
        "rear": [
            "Porsche car rear view photo",
            "Porsche back exterior photo",
            "Porsche 911 rear view",
            "Porsche Panamera back photo",
        ],
        "side": [
            "Porsche car side view photo",
            "Porsche lateral exterior profile",
            "Porsche 718 side view",
            "Porsche Cayenne side photo",
        ],
        "3quarter": [
            "Porsche car 3/4 front view photo",
            "Porsche three quarter angle exterior",
            "Porsche 911 three quarter",
            "Porsche front quarter view",
        ],
        "interior": [
            "Porsche car interior cockpit photo",
            "Porsche dashboard steering wheel",
            "Porsche cabin interior shot",
            "Porsche 911 interior photo",
        ],
    },
    "volkswagen": {
        "front": [
            "Volkswagen car front view photo",
            "VW Golf front exterior",
            "Volkswagen Passat front photo",
            "VW Tiguan front view",
        ],
        "rear": [
            "Volkswagen car rear view photo",
            "VW back exterior photo",
            "Volkswagen Golf rear view",
            "VW Polo back photo",
        ],
        "side": [
            "Volkswagen car side view photo",
            "VW lateral exterior profile",
            "Volkswagen Passat side view",
            "VW Golf side photo",
        ],
        "3quarter": [
            "Volkswagen car 3/4 front view photo",
            "VW three quarter angle exterior",
            "Volkswagen Golf three quarter",
            "VW front quarter view",
        ],
        "interior": [
            "Volkswagen car interior cockpit photo",
            "VW dashboard steering wheel",
            "VW cabin interior shot",
            "Volkswagen Golf interior photo",
        ],
    },
}

DEFAULT_VIEWS = ["front", "rear", "side", "3quarter", "interior"]

# Filtry Bing — ograniczamy do dużych zdjęć fotograficznych
BING_FILTERS = "+filterui:imagesize-large+filterui:photo-photo"


# ============================================================
# Pobieranie dla jednego widoku marki
# ============================================================

def collect_and_download_view(
    brand: str,
    view: str,
    per_view: int,
    output_dir: Path,
    workers: int,
) -> int:
    """
    Zbiera URL-e z Bing i pobiera obrazy dla jednej kombinacji marka+widok.
    Zwraca liczbę pobranych plików.
    """
    view_dir = output_dir / brand / view
    view_dir.mkdir(parents=True, exist_ok=True)

    # Sprawdź ile już mamy
    existing = len(list(view_dir.glob("*.jpg")) + list(view_dir.glob("*.jpeg"))
                   + list(view_dir.glob("*.png")) + list(view_dir.glob("*.webp")))
    if existing >= per_view:
        logger.info(f"  [{brand}/{view}] Już {existing} plików — pomijam")
        return existing

    still_needed = per_view - existing
    queries = VIEW_QUERIES.get(brand, {}).get(view, [f"{brand} car {view} view"])

    # Zbieramy URL-e — po kolejnych zapytaniach jeśli pierwsze nie wystarczy
    collected_urls: List[str] = []
    seen: set = set()
    for query in queries:
        if len(collected_urls) >= still_needed + 20:  # margines na błędy pobierania
            break
        urls = bing_search_urls(query, count=still_needed + 20, filters=BING_FILTERS)
        for u in urls:
            if u not in seen:
                seen.add(u)
                collected_urls.append(u)

    logger.info(f"  [{brand}/{view}] {len(collected_urls)} URL-ów → pobieranie...")

    downloaded = 0

    def _dl(url: str) -> bool:
        fname = hashlib.md5(url.encode()).hexdigest()[:16]
        ext = Path(url.split("?")[0]).suffix.lower() or ".jpg"
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            ext = ".jpg"
        dest = view_dir / f"{brand}_{view}_{fname}{ext}"
        return download_file(url, dest)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_dl, u) for u in collected_urls]
        for future in as_completed(futures):
            try:
                if future.result():
                    downloaded += 1
            except Exception:
                pass

    return existing + downloaded


# ============================================================
# Główna funkcja
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pobierz obrazy samochodów z Bing Images, posegregowane wg widoku"
    )
    parser.add_argument(
        "--output", "-o",
        default="tests/images/cars",
        help="Katalog wyjściowy (domyślnie: tests/images/cars)",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=200,
        help="Liczba zdjęć NA MARKĘ łącznie (domyślnie: 200, ~40 na widok)",
    )
    parser.add_argument(
        "--brands",
        nargs="+",
        choices=BRANDS,
        default=BRANDS,
        help="Marki do pobrania (domyślnie: wszystkie)",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        choices=DEFAULT_VIEWS,
        default=DEFAULT_VIEWS,
        help="Widoki do pobrania (domyślnie: front rear side 3quarter interior)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=5,
        help="Liczba równoległych pobierań na widok (domyślnie: 5)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_view = max(1, args.count // len(args.views))

    print(f"\n{'='*65}")
    print(f"  CarVision AI — Scraper samochodów (Bing Images)")
    print(f"{'='*65}")
    print(f"  Katalog wyjściowy : {output_dir.resolve()}")
    print(f"  Marki             : {', '.join(args.brands)}")
    print(f"  Widoki            : {', '.join(args.views)}")
    print(f"  Na markę łącznie  : {args.count}  (~{per_view} na widok)")
    print(f"  Workery           : {args.workers}")
    print(f"{'='*65}\n")

    grand_total = 0
    summary: Dict[str, Dict[str, int]] = {}

    for brand in args.brands:
        summary[brand] = {}
        brand_total = 0
        logger.info(f"=== Marka: {brand.upper()} ===")

        for view in args.views:
            n = collect_and_download_view(
                brand=brand,
                view=view,
                per_view=per_view,
                output_dir=output_dir,
                workers=args.workers,
            )
            summary[brand][view] = n
            brand_total += n
            logger.info(f"  ✓  {brand}/{view}: {n} plików")

        grand_total += brand_total
        logger.info(f"  >> {brand.upper()} łącznie: {brand_total} obrazów\n")

    # Podsumowanie
    print(f"\n{'='*65}")
    print(f"  PODSUMOWANIE")
    print(f"{'='*65}")
    header_views = "  ".join(f"{v:>9}" for v in args.views)
    print(f"  {'MARKA':<14} {header_views}   SUMA")
    print(f"  {'-'*62}")
    for brand, views_data in summary.items():
        cols = "  ".join(f"{views_data.get(v, 0):>9}" for v in args.views)
        total = sum(views_data.values())
        print(f"  {brand.upper():<14} {cols}   {total}")
    print(f"  {'-'*62}")
    all_files = (
        list(output_dir.rglob("*.jpg"))
        + list(output_dir.rglob("*.jpeg"))
        + list(output_dir.rglob("*.png"))
        + list(output_dir.rglob("*.webp"))
    )
    print(f"\n  Pliki na dysku  : {len(all_files)}")
    print(f"  Zapisano w      : {output_dir.resolve()}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
