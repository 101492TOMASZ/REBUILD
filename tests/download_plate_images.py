#!/usr/bin/env python3
"""
Scraper tablic rejestracyjnych â€” Bing Images, bez klucza API.

Pobiera zdjÄ™cia tablic z rĂłĹĽnych krajĂłw i widokĂłw:
    front_full   â€” caĹ‚y samochĂłd (tablica widoczna na przodzie)
    rear_full    â€” caĹ‚y samochĂłd (tablica na tyle)
    close_up     â€” zbliĹĽenie na tablicÄ™ (numer wyraĹşnie widoczny)
    street       â€” tablica w warunkach ulicznych

Struktura wyjĹ›ciowa:
    tests/images/plates/
        PL/
            front_full/
            rear_full/
            close_up/
            street/
        DE/
            ...
        GB/
            ...

UĹĽycie:
    python tests/download_plate_images.py
    python tests/download_plate_images.py --count 80 --countries PL DE GB
    python tests/download_plate_images.py --views close_up front_full
    python tests/download_plate_images.py --workers 6
"""

import sys
import hashlib
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))
from bing_scraper import bing_search_urls, download_file  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# Definicja krajĂłw + widokĂłw
# ============================================================

COUNTRIES = ["PL", "DE", "GB", "CZ", "FR", "IT"]

DEFAULT_VIEWS = ["close_up", "front_full", "rear_full", "street"]

# Filtr Bing â€” zdjÄ™cia fotograficzne, duĹĽe
BING_FILTERS = "+filterui:imagesize-large+filterui:photo-photo"

VIEW_QUERIES: Dict[str, Dict[str, List[str]]] = {
    "PL": {
        "close_up": [
            "Polish license plate close up photo",
            "tablica rejestracyjna Polska zbliĹĽenie foto",
            "Poland car registration plate macro",
            "Polish number plate detail photo",
        ],
        "front_full": [
            "Polish car front license plate photo",
            "Poland car front registration plate",
            "samochĂłd tablica rejestracyjna przĂłd",
            "Polish vehicle front number plate",
        ],
        "rear_full": [
            "Polish car rear license plate photo",
            "Poland car back registration plate",
            "samochĂłd tablica rejestracyjna tyĹ‚",
            "Polish vehicle rear number plate",
        ],
        "street": [
            "Polish car street license plate photo",
            "Poland vehicle registration plate road",
            "Polish car parking plate",
            "Poland street car plate photo",
        ],
    },
    "DE": {
        "close_up": [
            "German Kennzeichen close up photo",
            "deutsches Nummernschild Nahaufnahme",
            "Germany car registration plate macro",
            "German license plate detail photo",
        ],
        "front_full": [
            "German car front Kennzeichen photo",
            "Deutschland Auto Kennzeichen vorne",
            "Germany vehicle front number plate",
            "German car front registration photo",
        ],
        "rear_full": [
            "German car rear Kennzeichen photo",
            "Deutschland Auto Kennzeichen hinten",
            "Germany vehicle rear number plate",
            "German car back license plate photo",
        ],
        "street": [
            "German car street Kennzeichen photo",
            "Deutschland Nummernschild StraĂźe",
            "Germany vehicle plate road photo",
            "German car parking plate street",
        ],
    },
    "GB": {
        "close_up": [
            "UK number plate close up photo",
            "British license plate macro detail",
            "England car registration plate close",
            "UK DVLA number plate detail photo",
        ],
        "front_full": [
            "UK car front number plate photo",
            "British vehicle front registration",
            "England car front plate photo",
            "UK front number plate street",
        ],
        "rear_full": [
            "UK car rear number plate photo",
            "British vehicle rear registration",
            "England car back plate photo Yellow",
            "UK yellow rear number plate",
        ],
        "street": [
            "UK car street number plate photo",
            "British vehicle plate road",
            "England traffic number plate street",
            "UK car parking number plate photo",
        ],
    },
    "CZ": {
        "close_up": [
            "Czech SPZ license plate close up",
            "ÄŤeskĂˇ SPZ registraÄŤnĂ­ znaÄŤka detail",
            "Czech Republic registration plate macro",
            "CZ car plate close photo",
        ],
        "front_full": [
            "Czech car front SPZ plate photo",
            "Czech Republic vehicle front plate",
            "CZ car registration front photo",
        ],
        "rear_full": [
            "Czech car rear SPZ plate photo",
            "Czech Republic vehicle rear plate",
            "CZ car registration rear photo",
        ],
        "street": [
            "Czech car street plate photo",
            "CZ vehicle plate road photo",
            "Czech Republic car plate street",
        ],
    },
    "FR": {
        "close_up": [
            "French plaque immatriculation close up",
            "France car registration plate detail",
            "French number plate macro photo",
            "plaque immatriculation voiture France",
        ],
        "front_full": [
            "French car front plate photo",
            "France vehicle front registration",
            "voiture plaque avant France photo",
        ],
        "rear_full": [
            "French car rear plate photo",
            "France vehicle rear registration",
            "voiture plaque arriĂ¨re France",
        ],
        "street": [
            "French car street plate photo",
            "France vehicle plate road",
            "voiture rue plaque France",
        ],
    },
    "IT": {
        "close_up": [
            "Italian targa auto close up photo",
            "Italy car registration plate detail",
            "targa immatricolazione italiana macro",
            "Italian license plate close photo",
        ],
        "front_full": [
            "Italian car front targa photo",
            "Italy vehicle front registration",
            "auto targa anteriore Italia",
        ],
        "rear_full": [
            "Italian car rear targa photo",
            "Italy vehicle rear registration",
            "auto targa posteriore Italia",
        ],
        "street": [
            "Italian car street targa photo",
            "Italy vehicle plate road photo",
            "auto targa strada Italia",
        ],
    },
}


# ============================================================
# Pobieranie dla jednego widoku/kraju
# ============================================================

def collect_and_download_view(
    country: str,
    view: str,
    per_view: int,
    output_dir: Path,
    workers: int,
) -> int:
    view_dir = output_dir / country / view
    view_dir.mkdir(parents=True, exist_ok=True)

    existing = len(
        list(view_dir.glob("*.jpg"))
        + list(view_dir.glob("*.jpeg"))
        + list(view_dir.glob("*.png"))
        + list(view_dir.glob("*.webp"))
    )
    if existing >= per_view:
        logger.info(f"  [{country}/{view}] JuĹĽ {existing} plikĂłw â€” pomijam")
        return existing

    still_needed = per_view - existing
    queries = VIEW_QUERIES.get(country, {}).get(view, [f"{country} license plate {view}"])

    collected_urls: List[str] = []
    seen: set = set()
    for query in queries:
        if len(collected_urls) >= still_needed + 15:
            break
        urls = bing_search_urls(query, count=still_needed + 15, filters=BING_FILTERS)
        for u in urls:
            if u not in seen:
                seen.add(u)
                collected_urls.append(u)

    logger.info(f"  [{country}/{view}] {len(collected_urls)} URL-Ăłw â†’ pobieranie...")

    downloaded = 0

    def _dl(url: str) -> bool:
        fname = hashlib.md5(url.encode()).hexdigest()[:16]
        ext = Path(url.split("?")[0]).suffix.lower() or ".jpg"
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            ext = ".jpg"
        dest = view_dir / f"{country}_{view}_{fname}{ext}"
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
# GĹ‚Ăłwna funkcja
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pobierz zdjÄ™cia tablic rejestracyjnych z Bing Images"
    )
    parser.add_argument(
        "--output", "-o",
        default="tests/images/plates",
        help="Katalog wyjĹ›ciowy (domyĹ›lnie: tests/images/plates)",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=80,
        help="Liczba zdjÄ™Ä‡ NA KRAJ Ĺ‚Ä…cznie (domyĹ›lnie: 80, ~20 na widok)",
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        choices=COUNTRIES,
        default=COUNTRIES,
        help="Kraje (domyĹ›lnie: PL DE GB CZ FR IT)",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        choices=DEFAULT_VIEWS,
        default=DEFAULT_VIEWS,
        help="Widoki (domyĹ›lnie: close_up front_full rear_full street)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=5,
        help="RĂłwnolegĹ‚e pobierania (domyĹ›lnie: 5)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_view = max(1, args.count // len(args.views))

    print(f"\n{'='*65}")
    print(f"  CarVision AI â€” Scraper tablic rejestracyjnych (Bing Images)")
    print(f"{'='*65}")
    print(f"  Katalog wyjĹ›ciowy : {output_dir.resolve()}")
    print(f"  Kraje             : {', '.join(args.countries)}")
    print(f"  Widoki            : {', '.join(args.views)}")
    print(f"  Na kraj Ĺ‚Ä…cznie   : {args.count}  (~{per_view} na widok)")
    print(f"  Workery           : {args.workers}")
    print(f"{'='*65}\n")

    summary: Dict[str, Dict[str, int]] = {}

    for country in args.countries:
        summary[country] = {}
        logger.info(f"=== Kraj: {country} ===")

        for view in args.views:
            n = collect_and_download_view(
                country=country,
                view=view,
                per_view=per_view,
                output_dir=output_dir,
                workers=args.workers,
            )
            summary[country][view] = n
            logger.info(f"  âś“  {country}/{view}: {n} plikĂłw")

        country_total = sum(summary[country].values())
        logger.info(f"  >> {country} Ĺ‚Ä…cznie: {country_total}\n")

    # Podsumowanie
    print(f"\n{'='*65}")
    print(f"  PODSUMOWANIE")
    print(f"{'='*65}")
    header = "  ".join(f"{v:>11}" for v in args.views)
    print(f"  {'KRAJ':<8} {header}   SUMA")
    print(f"  {'-'*62}")
    for country, views_data in summary.items():
        cols = "  ".join(f"{views_data.get(v, 0):>11}" for v in args.views)
        total = sum(views_data.values())
        print(f"  {country:<8} {cols}   {total}")

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
