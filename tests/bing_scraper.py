#!/usr/bin/env python3
"""
Moduł pomocniczy — scraping Bing Images bez klucza API.

Eksportuje:
    bing_search_urls(query, count, filters) -> List[str]
    download_file(url, dest_path)            -> bool

Mechanizm:
    Bing zwraca JSON-w-HTML pod /images/async. Wyciągamy pole ``murl``
    (media URL — bezpośredni link do oryginału) przez regex.
    Rotacja User-Agent + losowe opóźnienia chronią przed blokadą.
"""

import re
import time
import random
import logging
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

BING_IMAGES_URL = "https://www.bing.com/images/async"
TIMEOUT = 20
MAX_FILE_MB = 12
BING_PAGE_SIZE = 35        # Bing zwraca max 35 wyników na stronę
DELAY_BETWEEN_PAGES = (1.2, 2.8)   # (min, max) sekund między stronami

# Rotacja User-Agent żeby Bing nie blokował
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) "
    "Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36",
]

# Regex wyciągający bezpośredni URL obrazka z odpowiedzi Bing
_MURL_RE = re.compile(r'"murl"\s*:\s*"(https?://[^"]+)"')
# Filtr rozszerzeń
_VALID_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def _random_ua() -> str:
    return random.choice(_USER_AGENTS)


def _build_bing_url(query: str, first: int, filters: str) -> str:
    params = {
        "q": query,
        "first": first,
        "count": BING_PAGE_SIZE,
        "tsc": "ImageHoverTitle",
        "adlt": "off",
        "qft": filters,
    }
    return BING_IMAGES_URL + "?" + urllib.parse.urlencode(params)


def _fetch_page(query: str, first: int, filters: str) -> List[str]:
    """Pobiera jedną stronę wyników Bing i zwraca listę URL-ów obrazków."""
    url = _build_bing_url(query, first, filters)
    headers = {
        "User-Agent": _random_ua(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": f"https://www.bing.com/images/search?q={urllib.parse.quote(query)}",
        "Connection": "keep-alive",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        logger.debug(f"Bing fetch error (first={first}): {e}")
        return []

    urls = _MURL_RE.findall(body)
    # Filtruj po rozszerzeniu
    return [u for u in urls if Path(urllib.parse.urlparse(u).path).suffix.lower() in _VALID_EXT]


def bing_search_urls(
    query: str,
    count: int = 200,
    filters: str = "+filterui:imagesize-large+filterui:photo-photo",
) -> List[str]:
    """
    Zbiera *count* URL-ów obrazków z Bing Images dla danego zapytania.
    Zwraca deduplikowaną listę (może być krótsza niż count jeśli Bing się skończy).
    """
    collected: List[str] = []
    seen: set = set()
    first = 1

    while len(collected) < count:
        page_urls = _fetch_page(query, first, filters)
        if not page_urls:
            logger.debug(f"Brak wyników dla '{query}' at first={first}")
            break

        added = 0
        for u in page_urls:
            if u not in seen:
                seen.add(u)
                collected.append(u)
                added += 1
                if len(collected) >= count:
                    break

        if added == 0:
            break  # Bing nie ma więcej unikalnych wyników

        first += BING_PAGE_SIZE
        delay = random.uniform(*DELAY_BETWEEN_PAGES)
        time.sleep(delay)

    return collected[:count]


def download_file(url: str, dest_path: Path) -> bool:
    """
    Pobiera plik z *url* do *dest_path*.
    Zwraca True jeśli sukces, False w innym wypadku.
    """
    if dest_path.exists() and dest_path.stat().st_size > 2000:
        return True  # Już pobrane

    headers = {
        "User-Agent": _random_ua(),
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": "https://www.bing.com/",
        "Accept-Encoding": "gzip, deflate",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            content = resp.read()

        if len(content) < 3000:  # Zbyt mały = placeholder / błąd
            return False
        if len(content) > MAX_FILE_MB * 1024 * 1024:
            return False

        dest_path.write_bytes(content)
        return True
    except urllib.error.HTTPError as e:
        logger.debug(f"HTTP {e.code}: {url}")
        return False
    except urllib.error.URLError as e:
        logger.debug(f"URLError: {e.reason} — {url}")
        return False
    except Exception as e:
        logger.debug(f"Błąd pobierania: {e} — {url}")
        return False
