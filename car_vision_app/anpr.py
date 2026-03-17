"""
Moduł ANPR (Automatic Number Plate Recognition).
Odpowiada za wykrywanie tablicy rejestracyjnej, wycinanie i odczyt znaków (OCR).
Obsługuje tablice europejskie (polskie, niemieckie, itp.)
"""

import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO
from typing import Tuple, Optional, List
import torch
import re
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ANPRModule:
    """Klasa do automatycznego rozpoznawania tablic rejestracyjnych."""
    
    def __init__(self, license_plate_model_path: str = None, gpu: bool = False):
        """
        Inicjalizacja modułu ANPR.
        
        Args:
            license_plate_model_path: Ścieżka do modelu wykrywania tablic (YOLO)
            gpu: Czy używać GPU dla OCR
        """
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
        
        # Inicjalizacja PaddleOCR
        try:
            logger.info("Initializing PaddleOCR...")
            device = 'gpu' if gpu else 'cpu'
            self.ocr_reader = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                show_log=False,
            )
            logger.info(f"✓ PaddleOCR initialized (device={device})")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise
        
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
        
        # Większy margines - 15% zamiast 5%
        margin_x = int((x2 - x1) * 0.15)
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
        Różne warianty dla lepszego rozpoznawania wszystkich części tablicy.
        """
        variants = []
        
        # Wariant 0: Oryginalny rozmiar z minimalnym preprocessingiem
        gray_orig = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced_orig = clahe.apply(gray_orig)
        variants.append(cv2.cvtColor(enhanced_orig, cv2.COLOR_GRAY2BGR))
        logger.debug(f"Added original size variant: {plate_crop.shape}")
        
        # 1. Powiększenie obrazu
        h, w = plate_crop.shape[:2]
        scale = max(4, 200 // h)  # Bardziej agresywne skalowanie
        enlarged = cv2.resize(plate_crop, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
        
        # Konwersja do szarości
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        
        # 2. Wyostrzanie
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2)
        sharpened = cv2.addWeighted(gray, 1.3, gaussian, -0.3, 0)
        
        # 3. CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(sharpened)
        
        # Wariant 1: Wyostrzony z CLAHE (najczęściej skuteczny)
        variants.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))
        
        # Wariant 2: Binaryzacja Otsu
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))
        
        # Wariant 3: Adaptacyjna binaryzacja
        adaptive = cv2.adaptiveThreshold(enhanced, 255, 
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        variants.append(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR))
        
        # Wariant 4: Oryginalny powiększony (fallback)
        variants.append(enlarged)
        
        # Wariant 5: Binaryzacja odwrócona (może pomóc jeśli tablica ma ciemne tło)
        _, otsu_inv = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        variants.append(cv2.cvtColor(otsu_inv, cv2.COLOR_GRAY2BGR))
        
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
            # Brytyjskie: MT62FPV (2 litery + 2 cyfry + 3 litery)
            (r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$', 100),
            # Niemieckie krótkie: WOBG404 (2-3 litery + 1 litera + 3-4 cyfry)
            (r'^[A-Z]{2,3}[A-Z]{1}[0-9]{3,4}$', 100),
            # Niemieckie standardowe: WOBAW642
            (r'^[A-Z]{1,3}[A-Z]{2}[0-9]{1,4}$', 100),
            # Polskie: WA12345, KR1234A
            (r'^[A-Z]{2,3}[0-9]{4,5}[A-Z]?$', 90),
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
    
    def read_plate_text(self, plate_crop: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Odczytuje tekst z obrazu tablicy rejestracyjnej.
        Próbuje wielu wariantów preprocessingu dla lepszych wyników.
        """
        # Generuj warianty obrazu
        variants = self.preprocess_plate_variants(plate_crop)
        logger.debug(f"Liczba wariantów preprocessingu: {len(variants)}")
        
        all_candidates = []
        
        # OCR na każdym wariancie
        for i, variant in enumerate(variants):
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
                    # Pomijaj fragmenty będące wyłącznie cyframi (np. "60", "80" = pas kraju EU/GB)
                    if len(cleaned) >= 2 and not cleaned.isdigit():
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
            logger.info(f"Selected best candidate: '{best_candidate['text']}'")
            return best_candidate['text'], best_candidate['score']
        
        logger.warning("No best candidate selected")
        return None, 0.0
    
    def select_best_candidate(self, candidates: List[dict]) -> Optional[dict]:
        """
        Wybiera najlepszy kandydat z listy rozpoznanych tekstów.
        Preferuje pełne tablice (więcej znaków) nad fragmentami.
        """
        if not candidates:
            return None
        
        logger.debug(f"All candidates: {[(c['text'], round(c['score'],2), c.get('is_valid')) for c in candidates]}")
        
        # Scoring: ważny jest format, długość tekstu i confidence
        def candidate_score(c):
            length = len(c['text'])
            # Preferuj tablice 6-9 znaków
            length_bonus = 2.0 if 6 <= length <= 9 else (1.0 if 5 <= length <= 10 else 0.3)
            valid_bonus = 1.5 if c.get('is_valid') else 1.0
            return c['score'] * length_bonus * valid_bonus
        
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
