"""
Moduł ANPR (Automatic Number Plate Recognition).
Odpowiada za wykrywanie tablicy rejestracyjnej, wycinanie i odczyt znaków (OCR).
Obsługuje tablice europejskie (polskie, niemieckie, itp.)
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Tuple, Optional, List
import torch
import re
import logging

logger = logging.getLogger(__name__)


class ANPRModule:
    """Klasa do automatycznego rozpoznawania tablic rejestracyjnych."""
    
    def __init__(self, license_plate_model_path: str = None, gpu: bool = None):
        """
        Inicjalizacja modułu ANPR.
        
        Args:
            license_plate_model_path: Ścieżka do modelu wykrywania tablic (YOLO)
            gpu: Czy używać GPU dla OCR (None = auto-detekcja)
        """
        if gpu is None:
            gpu = torch.cuda.is_available()
        # Domyślna ścieżka do modelu wykrywania tablic
        if license_plate_model_path is None:
            app_dir = os.path.dirname(os.path.abspath(__file__))
            base_path = os.path.dirname(app_dir)
            # Priorytet: anpr_best.pt w katalogu aplikacji, potem stary model
            candidates = [
                os.path.join(app_dir, 'anpr_best.pt'),
                os.path.join(base_path, 'Automatic-License-Plate-Recognition-using-YOLOv8-main', 'license_plate_detector.pt'),
            ]
            license_plate_model_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
            logger.info(f"Auto-selected plate model: {license_plate_model_path}")
        
        if not os.path.exists(license_plate_model_path):
            raise FileNotFoundError(f"Nie znaleziono modelu wykrywania tablic: {license_plate_model_path}")
        
        logger.info(f"►►► Loading plate detector: {license_plate_model_path}")
        self.plate_detector = YOLO(license_plate_model_path)
        logger.info(f"✔ Plate detector loaded: {os.path.basename(license_plate_model_path)}")
        
        # Leniwa inicjalizacja PaddleOCR — import i tworzenie dopiero przy pierwszym użyciu
        self._gpu = gpu
        self._ocr_reader = None
        
        # Kody krajów EU/EEA na niebieskim pasie tablicy — do ignorowania
        self.EU_COUNTRY_CODES = {
            'A', 'AL', 'AND', 'ARM', 'AZ', 'B', 'BG', 'BIH', 'BY', 'CH',
            'CY', 'CZ', 'D', 'DK', 'E', 'EST', 'F', 'FIN', 'FL', 'FR',
            'GB', 'GBG', 'GBJ', 'GBM', 'GBZ', 'GE', 'GR', 'H', 'HR',
            'HU', 'I', 'IRL', 'IS', 'KOS', 'L', 'LT', 'LV', 'M', 'MC',
            'MD', 'ME', 'MK', 'MNE', 'N', 'NL', 'NMK', 'P', 'PL', 'RKS',
            'RO', 'RSM', 'RUS', 'S', 'SK', 'SLO', 'SRB', 'TR', 'UA', 'V',
        }

        # Mapowanie podobnych znaków dla korekcji OCR
        self.similar_chars = {
            '0': 'O', 'O': '0',
            '1': 'I', 'I': '1', 'L': '1',
            '2': 'Z', 'Z': '2',
            '4': 'A', 'A': '4',
            '5': 'S', 'S': '5',
            '6': 'G', 'G': '6',
            '8': 'B', 'B': '8',
            'D': '0', 'Q': '0',
        }
    
    @property
    def ocr_reader(self):
        """Leniwa inicjalizacja PaddleOCR — import i tworzenie przy pierwszym użyciu."""
        if self._ocr_reader is None:
            from paddleocr import PaddleOCR
            logger.info("Initializing PaddleOCR (lazy)...")
            self._ocr_reader = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                show_log=False,
                use_gpu=self._gpu,
                det_db_thresh=0.3,
                det_db_box_thresh=0.45,
                rec_batch_num=8,
            )
            logger.info(f"✓ PaddleOCR initialized (gpu={self._gpu})")
        return self._ocr_reader

    def detect_license_plate(self, image: np.ndarray) -> Tuple[Optional[list], float]:
        """
        Wykrywa tablicę rejestracyjną na obrazie.
        Niższy threshold dla lepszego wykrywania.
        """
        try:
            results = self.plate_detector(image, conf=0.3)[0]  # Niższy threshold
            logger.debug(f"Liczba wykrytych detections: {len(results.boxes.data.tolist())}")
            
            best_plate = None
            best_score = 0.0
            
            for detection in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                logger.debug(f"Detection: score={score:.3f}, bbox=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
                if score > best_score:
                    best_score = score
                    best_plate = [int(x1), int(y1), int(x2), int(y2)]
            
            logger.info(f"Best plate detection: score={best_score:.3f}, bbox={best_plate}")
            return best_plate, best_score
        except Exception as e:
            logger.error(f"Error in detect_license_plate: {str(e)}")
            return None, 0.0
    
    def crop_license_plate(self, image: np.ndarray, bbox: list) -> np.ndarray:
        """Wycina tablicę rejestracyjną z obrazu z większym marginesem."""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Większy margines horyzontalny (20%) — nie obcinaj skrajnych liter
        margin_x = int((x2 - x1) * 0.20)
        margin_y = int((y2 - y1) * 0.15)
        
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)
    
        logger.debug(f"Crop bbox: ({x1}, {y1}, {x2}, {y2}), original: {bbox}")
        return image[y1:y2, x1:x2].copy()
    def preprocess_plate_variants(self, plate_crop: np.ndarray) -> List[np.ndarray]:
        """
        Tworzy warianty przetworzenia obrazu tablicy dla OCR.
        4 warianty: surowy, wyostrzony+CLAHE, Otsu, adaptacyjny.
        """
        variants = []
        h, w = plate_crop.shape[:2]

        # Wyższe skalowanie — 300px wysokości zamiast 200
        scale = max(4, 300 // max(h, 1))
        enlarged = cv2.resize(plate_crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        # Wariant 0: Surowy powiększony (bez filtrów — PaddleOCR sam sobie radzi)
        variants.append(enlarged.copy())

        # Konwersja do szarości
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)

        # Wyostrzanie
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2)
        sharpened = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(sharpened)

        # Wariant 1: Wyostrzony z CLAHE
        variants.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))

        # Wariant 2: Binaryzacja Otsu
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))

        # Wariant 3: Adaptacyjny threshold (dobry na nierówne oświetlenie)
        adaptive = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10
        )
        # Morfologia — zamknij przerwy w glifach
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
        variants.append(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR))

        return variants
    
    def clean_plate_text(self, text: str) -> str:
        """Czyści tekst tablicy z niepotrzebnych znaków."""
        # Usuń wszystko oprócz liter i cyfr
        text = text.upper()
        text = re.sub(r'[^A-Z0-9]', '', text)
        return text
    
    def correct_ocr_errors(self, text: str) -> str:
        """
        Koryguje typowe błędy OCR próbując różnych wariantów pierwszego znaku
        i wybierając ten który najlepiej pasuje do formatu tablicy.
        """
        if len(text) < 4:
            return text
        
        logger.debug(f"Correcting OCR errors for: '{text}'")
        
        # H jest mylone z W oraz M — generuj alternatywy
        # Alternatywy testujemy PRZED oryginałem, żeby przy równym score preferować korektę
        candidates_to_try = []
        if text[0] == 'H':
            candidates_to_try.append('W' + text[1:])  # W jest częstsze niż H w kodach regionów
            candidates_to_try.append('M' + text[1:])
        elif text[0] == 'W':
            candidates_to_try.append('M' + text[1:])
        elif text[0] == 'M':
            candidates_to_try.append('W' + text[1:])
        candidates_to_try.append(text)  # oryginał na końcu — wygrywa tylko gdy ma wyższy score
        
        best_result = text
        best_score = -1
        for candidate in candidates_to_try:
            corrected = self._apply_char_corrections(candidate)
            _, score = self.validate_plate_format(corrected)
            if score >= best_score:  # >= — preferuj pierwszego kandydata z max score (W przed H)
                best_score = score
                best_result = corrected
        
        logger.debug(f"Correction result: '{best_result}' (format_score={best_score})")
        return best_result
    
    def _apply_char_corrections(self, text: str) -> str:
        """Stosuje korekcje znaków OCR w oparciu o pozycję cyfr/liter."""
        corrected = list(text)
        
        # Znajdź ciągłe bloki cyfr
        digit_blocks = []
        i = 0
        while i < len(corrected):
            if corrected[i].isdigit():
                start = i
                while i < len(corrected) and corrected[i].isdigit():
                    i += 1
                digit_blocks.append((start, i))
            else:
                i += 1
        
        # Popraw cyfry na litery przed pierwszym blokiem cyfr (prefix = kod regionu)
        first_digit_pos = digit_blocks[0][0] if digit_blocks else len(corrected)
        digit_to_letter = {'0': 'O', '1': 'I', '2': 'Z', '3': 'E', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}
        for i in range(min(first_digit_pos, 3)):
            if corrected[i].isdigit() and corrected[i] in digit_to_letter:
                corrected[i] = digit_to_letter[corrected[i]]
        
        # Popraw litery na cyfry w ostatnim bloku TYLKO gdy nic po nim nie ma (format DE/PL)
        # NIE koryguj gdy są litery po cyfrach (format UK: MT62FPV)
        if digit_blocks:
            last_start, last_end = digit_blocks[-1]
            has_letters_after = any(c.isalpha() for c in corrected[last_end:])
            if not has_letters_after:
                letter_to_digit = {
                    'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1',
                    'Z': '2', 'E': '3', 'A': '4', 'S': '5',
                    'G': '6', 'T': '7', 'B': '8'
                }
                for i in range(last_start, last_end):
                    if corrected[i].isalpha() and corrected[i] in letter_to_digit:
                        corrected[i] = letter_to_digit[corrected[i]]
        
        return ''.join(corrected)
    
    def validate_plate_format(self, text: str) -> Tuple[bool, int]:
        """
        Sprawdza czy tekst pasuje do formatu europejskiej tablicy.
        Zwraca (czy_valid, score).
        """
        if len(text) < 4 or len(text) > 10:  # Rozszerzony range
            return False, 0
        
        # Wzorce tablic rejestracyjnych (od najbardziej specyficznych)
        patterns = [
            # Polskie nowe: 2 litery + 5 cyfr (np. WA12345)
            (r'^[A-Z]{2}[0-9]{5}$', 100),
            # Polskie nowe: 3 litery + 5 cyfr (np. FMI66259, DPL12345)
            (r'^[A-Z]{3}[0-9]{5}$', 100),
            # Polskie nowe: 2 litery + 4 cyfry + 1 litera (np. WI1234A)
            (r'^[A-Z]{2}[0-9]{4}[A-Z]$', 100),
            # Polskie nowe: 3 litery + 4 cyfry + 1 litera (np. FMI1234A)
            (r'^[A-Z]{3}[0-9]{4}[A-Z]$', 100),
            # Polskie nowe: 2-3 litery + 4 cyfry (np. KR1234)
            (r'^[A-Z]{2,3}[0-9]{4}$', 95),
            # Brytyjskie: MT62FPV (2 litery + 2 cyfry + 3 litery)
            (r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$', 100),
            # Niemieckie krótkie: WOBG404 (2-3 litery + 1 litera + 3-4 cyfry)
            (r'^[A-Z]{2,3}[A-Z]{1}[0-9]{3,4}$', 100),
            # Niemieckie standardowe: WOBAW642
            (r'^[A-Z]{1,3}[A-Z]{2}[0-9]{1,4}$', 100),
            # Ogólne europejskie z cyframi
            (r'^[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,3}$', 70),
        ]
        
        for pattern, score in patterns:
            if re.match(pattern, text):
                return True, score
        
        # Częściowe dopasowanie - jeśli wygląda jak tablica
        if len(text) >= 4 and any(c.isdigit() for c in text) and any(c.isalpha() for c in text):
            return True, 40
        
        return False, 0
    
    def remove_duplicate_chars(self, text: str) -> str:
        """
        Usuwa podejrzane duplikaty znaków które mogą być artefaktami OCR.
        Na przykład: WOBSG404 -> WOBG404 (usunięcie fałszywego S)
        """
        if len(text) <= 6:
            return text
        
        # Dla niemieckich tablic sprawdź wzorzec
        # Format: 1-3 litery (miasto) + 1-2 litery + 1-4 cyfry
        
        # Znajdź gdzie zaczynają się cyfry
        digit_start = len(text)
        for i, c in enumerate(text):
            if c.isdigit():
                digit_start = i
                break
        
        # Część literowa przed cyframi
        letters_part = text[:digit_start]
        digits_part = text[digit_start:]
        
        # Typowe długości niemieckich tablic
        # Krótkie: 3+1 = 4 litery (WOB+G)
        # Standardowe: 3+2 = 5 liter (WOB+AW), 2+2 = 4 (WI+AB), 1+2 = 3 (B+AB)
        
        # Jeśli mamy za dużo liter, spróbuj usunąć szum
        if len(letters_part) > 5:
            # Próbuj usunąć pojedyncze znaki
            for i in range(3, len(letters_part) - 1):  # Nie usuwaj z pierwszych 3 (kod miasta) ani ostatniego
                candidate = letters_part[:i] + letters_part[i+1:] + digits_part
                is_valid, score = self.validate_plate_format(candidate)
                if is_valid and score >= 90:
                    return candidate
        
        # Specjalny przypadek: WOBSG404 -> szukamy wzorca gdzie 4-ta lub 5-ta litera jest szumem
        # Typowe wzorce niemieckie: XXX+Y+1234 lub XX+YY+1234
        if len(letters_part) == 5 and len(digits_part) >= 3:
            # Spróbuj XXXZ+digits (3+1) - usuń 4-tą literę
            candidate1 = letters_part[:3] + letters_part[4] + digits_part
            # Spróbuj XXYZ+digits (2+2) - usuń 3-cią literę
            candidate2 = letters_part[:2] + letters_part[3:] + digits_part
            
            for candidate in [candidate1, candidate2]:
                is_valid, score = self.validate_plate_format(candidate)
                if is_valid and score >= 90:
                    return candidate
        
        # Jeśli mamy dokładnie 4 litery i 3+ cyfr, sprawdź format
        if len(letters_part) == 4 and len(digits_part) >= 3:
            # Może być poprawne (WOB+G) lub szum
            # Spróbuj usunąć 4-tą literę (wzorzec 3+1 vs szum)
            pass  # Aktualny format może być poprawny
        
        return text
    
    def _remove_eu_blue_strip(self, plate_crop: np.ndarray) -> np.ndarray:
        """
        Wykrywa i usuwa niebieski pas z kodem kraju EU z lewej strony tablicy.
        Konserwatywne podejście — lepiej zostawić trochę niebieskiego niż obciąć literę.
        """
        try:
            h, w = plate_crop.shape[:2]
            # Sprawdzamy maks. lewą 15% obrazu (nie 25% — zbyt agresywne)
            max_strip = max(1, int(w * 0.15))
            hsv = cv2.cvtColor(plate_crop[:, :max_strip], cv2.COLOR_BGR2HSV)
            blue_mask = cv2.inRange(
                hsv,
                np.array([95, 60, 40], dtype=np.uint8),
                np.array([145, 255, 255], dtype=np.uint8),
            )
            # Znajdź ostatnią kolumnę z dominującym niebieskim (>40% pikseli)
            strip_end = 0
            gap_count = 0
            for col_idx in range(max_strip):
                blue_ratio = np.sum(blue_mask[:, col_idx] > 0) / h
                if blue_ratio > 0.40:
                    strip_end = col_idx + 1
                    gap_count = 0
                else:
                    gap_count += 1
                    # Koniec pasa — 3 kolumny bez niebieskiego
                    if gap_count >= 3 and strip_end > 0:
                        break
            # Cofnij o 4px bufor — nie obcinaj glifów stykających się z pasem
            strip_end = max(0, strip_end - 4)
            if strip_end > 0:
                logger.debug(f"Usunięto niebieski pas EU: {strip_end}px z lewej (img_w={w})")
                return plate_crop[:, strip_end:]
        except Exception as e:
            logger.debug(f"Błąd usuwania niebieskiego pasa EU: {e}")
        return plate_crop

    def _strip_eu_country_prefix(self, text: str) -> str:
        """
        Usuwa prefiks kodu kraju EU z rozpoznanego tekstu tablicy.
        Np. "PLSC6271X" -> "SC6271X".
        UWAGA: Nie stripuje jednoliterowych kodów (F, D, I, S, N, B, E, H, L, M, V itd.)
        bo zbyt często pokrywają się z pierwszą literą kodu regionu tablicy.
        """
        for length in (3, 2):  # tylko 2-3 literowe kody — jednoliterowe pomijamy
            if len(text) > length + 4:  # musi zostać co najmniej 5 znaków
                prefix = text[:length]
                remainder = text[length:]
                if prefix in self.EU_COUNTRY_CODES:
                    # Upewnij się, że reszta wygląda jak tablica
                    if remainder and remainder[0].isalpha():
                        # Dodatkowa walidacja: reszta musi być lepsza niż całość
                        _, score_full = self.validate_plate_format(text)
                        _, score_rest = self.validate_plate_format(remainder)
                        if score_rest > score_full:
                            logger.debug(f"Usunięto prefiks kraju '{prefix}' z '{text}' -> '{remainder}'")
                            return remainder
        return text

    def read_plate_text(self, plate_crop: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Odczytuje tekst z obrazu tablicy rejestracyjnej.
        Próbuje wielu wariantów preprocessingu dla lepszych wyników.
        Uruchamia OCR zarówno na oryginale jak i po usunięciu pasa EU.
        """
        plate_no_strip = self._remove_eu_blue_strip(plate_crop)

        # Dwa zestawy wariantów: z oryginalnym cropem i bez pasa EU
        all_variants = []
        all_variants.extend(self.preprocess_plate_variants(plate_no_strip))
        # Jeśli usunięto pas, dodaj też oryginał (zabezpieczenie przed zbyt agresywnym cięciem)
        if plate_no_strip is not plate_crop:
            all_variants.extend(self.preprocess_plate_variants(plate_crop))

        logger.debug(f"Liczba wariantów preprocessingu: {len(all_variants)}")
        
        all_candidates = []
        
        # OCR na każdym wariancie
        for i, variant in enumerate(all_variants):
            try:
                results = self.ocr_reader.ocr(variant, cls=True)
                logger.debug(f"Wariant {i}: raw results type={type(results)}")

                # PaddleOCR 2.x zwraca List[List[List]] lub [[None]]
                if not results or results[0] is None:
                    logger.debug(f"Wariant {i}: brak wyników")
                    continue

                detections = results[0]  # pierwsza strona
                logger.debug(f"Wariant {i}: {len(detections)} detection(s)")

                # Sortuj fragmenty od lewej do prawej po pozycji X bounding boxa
                detections_sorted = sorted(detections, key=lambda r: r[0][0][0] if r[0] else 0)

                # Dodaj fragmenty jako oddzielne kandydaty
                variant_texts = []
                variant_scores = []
                for detection in detections_sorted:
                    bbox, (text, score) = detection[0], detection[1]
                    cleaned = self.clean_plate_text(text)
                    logger.debug(f"  OCR fragment: '{text}' -> '{cleaned}' (score={score:.3f})")
                    # Do łączenia: akceptuj wszystkie fragmenty >= 1 znaku (także same cyfry)
                    # ale odfiltruj kody krajów EU (PL, D, etc.)
                    if len(cleaned) >= 1 and cleaned not in self.EU_COUNTRY_CODES:
                        variant_texts.append(cleaned)
                        variant_scores.append(score)

                    # Dodaj też każdy fragment z >= 4 znaki osobno
                    if len(cleaned) >= 4:
                        corrected = self.correct_ocr_errors(cleaned)
                        corrected = self.remove_duplicate_chars(corrected)
                        is_valid, format_score = self.validate_plate_format(corrected)
                        all_candidates.append({
                            'text': corrected,
                            'score': score,
                            'original': text,
                            'format_score': format_score,
                            'is_valid': is_valid
                        })

                # Kluczowy krok: połącz fragmenty sortując od lewej do prawej (po X)
                if len(variant_texts) >= 2:
                    combined_text = ''.join(variant_texts)
                    combined_score = sum(variant_scores) / len(variant_scores)
                    logger.debug(f"  Combined: '{combined_text}' (avg_score={combined_score:.3f})")
                    if len(combined_text) >= 4:
                        corrected = self.correct_ocr_errors(combined_text)
                        corrected = self.remove_duplicate_chars(corrected)
                        is_valid, format_score = self.validate_plate_format(corrected)
                        # Preferuj połączone wyniki - daj bonus do score
                        all_candidates.append({
                            'text': corrected,
                            'score': min(combined_score + 0.1, 1.0),
                            'original': combined_text,
                            'format_score': format_score,
                            'is_valid': is_valid
                        })
                        logger.debug(f"  Combined corrected: '{corrected}', valid={is_valid}")

            except Exception as e:
                logger.error(f"Error in OCR variant {i}: {str(e)}")
                continue
        
        logger.info(f"Total OCR candidates: {len(all_candidates)}")
        if not all_candidates:
            logger.warning("No valid OCR candidates found")
            return None, 0.0
        
        # Grupuj podobne wyniki i wybierz najlepszy
        best_candidate = self.select_best_candidate(all_candidates)
        
        if best_candidate:
            final_text = self._strip_eu_country_prefix(best_candidate['text'])
            logger.info(f"Selected best candidate: '{best_candidate['text']}' -> '{final_text}'")
            return final_text, best_candidate['score']
        
        logger.warning("No best candidate selected")
        return None, 0.0
    
    def select_best_candidate(self, candidates: List[dict]) -> Optional[dict]:
        """
        Wybiera najlepszy kandydat z listy rozpoznanych tekstów.
        Preferuje pełne tablice z pasującym formatem.
        """
        if not candidates:
            return None

        logger.debug(f"All candidates: {[(c['text'], round(c['score'],2), c.get('format_score',0), c.get('is_valid')) for c in candidates]}")

        def candidate_score(c):
            length = len(c['text'])
            # Preferuj tablice 6-9 znaków, akceptuj 5 i 10
            if 6 <= length <= 9:
                length_bonus = 2.0
            elif length == 5 or length == 10:
                length_bonus = 1.3
            else:
                length_bonus = 0.3
            # Mocny bonus za pasujący format tablicy
            fmt = c.get('format_score', 0)
            format_bonus = 1.0 + fmt / 50.0  # 100 format_score → 3.0x
            valid_bonus = 1.3 if c.get('is_valid') else 1.0
            return c['score'] * length_bonus * format_bonus * valid_bonus

        sorted_candidates = sorted(candidates, key=candidate_score, reverse=True)
        logger.debug(f"Top candidates: {[(c['text'], round(candidate_score(c),3)) for c in sorted_candidates[:5]]}")

        return sorted_candidates[0]
    
    def format_european_plate(self, text: str) -> str:
        """
        Formatuje tekst jako europejską tablicę rejestracyjną.
        Dodaje myślnik/spację w odpowiednim miejscu.
        
        Niemieckie tablice: WOB AW 642 -> WOB-AW642
        Polskie tablice: WA 12345 -> WA 12345
        """
        if len(text) < 5:
            return text
        
        # Niemieckie tablice: 1-3 litery (miasto) + 1-2 litery + 1-4 cyfry
        # Wzorzec: znajdź gdzie kończy się kod miasta (1-3 litery na początku)
        
        # Znajdź prefiks literowy (kod miasta)
        city_end = 0
        for i, char in enumerate(text):
            if char.isalpha() and i < 3:
                city_end = i + 1
            else:
                break
        
        if city_end >= 1:
            city_code = text[:city_end]
            rest = text[city_end:]
            
            if rest and len(rest) >= 3:
                return f"{city_code}-{rest}"
        
        return text
    
    def process(self, car_crop: np.ndarray) -> dict:
        """
        Główna metoda przetwarzania - wykrywa i odczytuje tablicę rejestracyjną.
        """
        logger.info(f"Processing car crop of size {car_crop.shape}")
        
        result = {
            'detected': False,
            'bbox': None,
            'plate_crop': None,
            'text': "Nie wykryto tablicy",
            'confidence': 0.0
        }
        
        # Wykryj tablicę
        bbox, detect_conf = self.detect_license_plate(car_crop)
        
        if bbox is None:
            logger.warning("No license plate detected")
            return result
        
        logger.info(f"Plate detected with confidence {detect_conf:.3f}")
        result['bbox'] = bbox
        result['detected'] = True
        
        # Wytnij tablicę
        plate_crop = self.crop_license_plate(car_crop, bbox)
        result['plate_crop'] = plate_crop
        logger.debug(f"Plate crop size: {plate_crop.shape}")
        
        # Odczytaj tekst
        text, ocr_conf = self.read_plate_text(plate_crop)
        
        if text:
            # Formatuj jako europejską tablicę
            formatted_text = self.format_european_plate(text)
            result['text'] = formatted_text
            result['confidence'] = ocr_conf
            logger.info(f"Plate text recognized: '{formatted_text}' (confidence: {ocr_conf:.3f})")
        else:
            logger.warning("Failed to read plate text")
        
        return result
    
    def draw_plate_bbox(self, image: np.ndarray, bbox: list, text: str = None) -> np.ndarray:
        """Rysuje bounding box tablicy na obrazie."""
        annotated = image.copy()
        x1, y1, x2, y2 = bbox
        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        if text:
            cv2.putText(annotated, text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return annotated
