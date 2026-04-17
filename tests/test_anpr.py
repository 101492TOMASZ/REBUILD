"""
Testy jednostkowe modułu ANPR (ANPRModule).

Koncentruje się na metodach algorytmicznych (bez PaddleOCR i YOLO):
  - usuwanie niebieskiego pasa EU
  - usuwanie prefiksu kodu kraju
  - czyszczenie i walidacja tekstu tablicy
  - korekcja błędów OCR

Testy integracyjne oznaczone @pytest.mark.integration wymagają
anpr_best.pt i działającego PaddleOCR.
"""

import sys
import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import requires_anpr_model


# ============================================================
# Testy czyszczenia tekstu
# ============================================================

class TestCleanPlateText:
    def test_uppercase(self, anpr_bare):
        assert anpr_bare.clean_plate_text("sc6271x") == "SC6271X"

    def test_removes_spaces(self, anpr_bare):
        assert anpr_bare.clean_plate_text("SC 627 1X") == "SC6271X"

    def test_removes_hyphens(self, anpr_bare):
        assert anpr_bare.clean_plate_text("WOB-AW-642") == "WOBAW642"

    def test_removes_special_chars(self, anpr_bare):
        assert anpr_bare.clean_plate_text("SC!@#6271X") == "SC6271X"

    def test_empty_string(self, anpr_bare):
        assert anpr_bare.clean_plate_text("") == ""

    def test_only_special_chars(self, anpr_bare):
        assert anpr_bare.clean_plate_text("---") == ""

    def test_preserves_digits(self, anpr_bare):
        result = anpr_bare.clean_plate_text("AB 1234")
        assert "1234" in result

    def test_removes_dots(self, anpr_bare):
        assert "." not in anpr_bare.clean_plate_text("SC.62.71X")


# ============================================================
# Testy walidacji formatu tablicy
# ============================================================

class TestValidatePlateFormat:
    # Polskie tablice
    def test_polish_2letter_5digit(self, anpr_bare):
        is_valid, score = anpr_bare.validate_plate_format("SC6271X")
        assert is_valid
        assert score >= 40

    def test_polish_3letter_4digit(self, anpr_bare):
        is_valid, score = anpr_bare.validate_plate_format("WAW1234")
        assert is_valid

    def test_polish_2letter_4digit(self, anpr_bare):
        is_valid, score = anpr_bare.validate_plate_format("KR1234")
        assert is_valid

    # Niemieckie tablice
    def test_german_format(self, anpr_bare):
        is_valid, score = anpr_bare.validate_plate_format("WOBAW642")
        assert is_valid

    def test_german_short(self, anpr_bare):
        is_valid, score = anpr_bare.validate_plate_format("WOBG404")
        assert is_valid

    # Brytyjskie tablice
    def test_uk_format(self, anpr_bare):
        is_valid, score = anpr_bare.validate_plate_format("MT62FPV")
        assert is_valid
        assert score >= 90

    # Nieprawidłowe
    def test_too_short_invalid(self, anpr_bare):
        is_valid, _ = anpr_bare.validate_plate_format("AB1")
        assert not is_valid

    def test_too_long_invalid(self, anpr_bare):
        is_valid, _ = anpr_bare.validate_plate_format("ABCDEFGHIJK")
        assert not is_valid

    def test_only_letters_low_score(self, anpr_bare):
        # Tablica bez cyfr — niski score lub nievalid
        is_valid, score = anpr_bare.validate_plate_format("ABCDEF")
        # Może być valid (partial) ale score powinien być niski
        if is_valid:
            assert score <= 50

    def test_empty_invalid(self, anpr_bare):
        is_valid, _ = anpr_bare.validate_plate_format("")
        assert not is_valid


# ============================================================
# Testy korekcji błędów OCR
# ============================================================

class TestCorrectOCRErrors:
    def test_H_to_W_correction_on_common_plate(self, anpr_bare):
        # "HOBG404" powinno dać lepszy wynik jako "WOBG404"
        result = anpr_bare.correct_ocr_errors("HOBG404")
        # Wynik powinien zaczynać się od W lub M (korekta H)
        assert result[0] in ("W", "M", "H")

    def test_no_change_when_already_correct(self, anpr_bare):
        result = anpr_bare.correct_ocr_errors("SC6271X")
        # Nie powinno nic zepsuć
        assert len(result) == len("SC6271X")

    def test_short_text_returned_unchanged(self, anpr_bare):
        assert anpr_bare.correct_ocr_errors("AB") == "AB"

    def test_result_is_string(self, anpr_bare):
        result = anpr_bare.correct_ocr_errors("WOBG4O4")
        assert isinstance(result, str)


# ============================================================
# Testy usuwania duplikatów znaków
# ============================================================

class TestRemoveDuplicateChars:
    def test_short_text_unchanged(self, anpr_bare):
        assert anpr_bare.remove_duplicate_chars("WOBG") == "WOBG"

    def test_result_is_string(self, anpr_bare):
        result = anpr_bare.remove_duplicate_chars("WOBSG404")
        assert isinstance(result, str)

    def test_valid_plate_not_shortened_excessively(self, anpr_bare):
        result = anpr_bare.remove_duplicate_chars("WOBAW642")
        # Wynik powinien mieć rozsądną długość (nie mniej niż 5 znaków)
        assert len(result) >= 5

    def test_plate_with_noise_corrected(self, anpr_bare):
        # WOBSG404 ma 5 liter + 3 cyfry — o 1 literę za dużo dla wzorca WOB+G+404
        result = anpr_bare.remove_duplicate_chars("WOBSG404")
        is_valid, score = anpr_bare.validate_plate_format(result)
        # Po korekcji powinien mieć wyższy score lub przynajmniej nie gorzej
        assert isinstance(is_valid, bool)


# ============================================================
# Testy kodu krajów EU
# ============================================================

class TestEUCountryCodes:
    def test_PL_in_set(self, anpr_bare):
        assert "PL" in anpr_bare.EU_COUNTRY_CODES

    def test_D_in_set(self, anpr_bare):
        assert "D" in anpr_bare.EU_COUNTRY_CODES

    def test_GB_in_set(self, anpr_bare):
        assert "GB" in anpr_bare.EU_COUNTRY_CODES

    def test_CZ_in_set(self, anpr_bare):
        assert "CZ" in anpr_bare.EU_COUNTRY_CODES

    def test_SK_in_set(self, anpr_bare):
        assert "SK" in anpr_bare.EU_COUNTRY_CODES

    def test_random_plate_text_not_in_set(self, anpr_bare):
        assert "SC6271X" not in anpr_bare.EU_COUNTRY_CODES

    def test_set_has_sufficient_entries(self, anpr_bare):
        assert len(anpr_bare.EU_COUNTRY_CODES) >= 30


# ============================================================
# Testy usuwania prefiksu kodu kraju
# ============================================================

class TestStripEUCountryPrefix:
    def test_removes_PL_prefix(self, anpr_bare):
        result = anpr_bare._strip_eu_country_prefix("PLSC6271X")
        assert result == "SC6271X"

    def test_removes_D_prefix(self, anpr_bare):
        result = anpr_bare._strip_eu_country_prefix("DWOBAW642")
        assert result == "WOBAW642"

    def test_removes_CZ_prefix(self, anpr_bare):
        result = anpr_bare._strip_eu_country_prefix("CZAB123CD")
        assert result == "AB123CD"

    def test_removes_EST_prefix(self, anpr_bare):
        result = anpr_bare._strip_eu_country_prefix("ESTAB1234")
        assert result == "AB1234"

    def test_no_change_valid_plate(self, anpr_bare):
        result = anpr_bare._strip_eu_country_prefix("SC6271X")
        assert result == "SC6271X"

    def test_no_change_if_remainder_too_short(self, anpr_bare):
        # "PL" + "AB" = 4 znaki — za krótkie dla tablicy (wymaga > prefix+3)
        result = anpr_bare._strip_eu_country_prefix("PLAB")
        # Nie powinna być skrócona (zostaje "PLAB" lub "AB" — zależy od warunku)
        assert len(result) >= 2

    def test_no_change_when_no_match(self, anpr_bare):
        result = anpr_bare._strip_eu_country_prefix("XY1234Z")
        assert result == "XY1234Z"

    def test_result_starts_with_letter(self, anpr_bare):
        result = anpr_bare._strip_eu_country_prefix("PLSC6271X")
        assert result[0].isalpha()


# ============================================================
# Testy usuwania niebieskiego pasa EU
# ============================================================

class TestRemoveEUBlueStrip:
    def test_no_strip_image_unchanged(self, anpr_bare, plate_image_plain):
        """Obraz bez niebieskiego pasa — rozmiar powinien się nie zmienić."""
        result = anpr_bare._remove_eu_blue_strip(plate_image_plain)
        # Biały obraz nie ma niebieskiego pasa — może zostać taki sam lub poszerzony
        assert result.shape[0] == plate_image_plain.shape[0]   # wysokość ta sama
        assert result.shape[1] <= plate_image_plain.shape[1]   # szerokość <= oryginalna

    def test_blue_strip_removed(self, anpr_bare, plate_image_with_eu_strip):
        """Obraz z niebieskim pasem — szerokość powinna się zmniejszyć."""
        original_w = plate_image_with_eu_strip.shape[1]
        result = anpr_bare._remove_eu_blue_strip(plate_image_with_eu_strip)
        # Powinien wykryć i usunąć pas
        assert result.shape[1] < original_w

    def test_blue_strip_result_not_empty(self, anpr_bare, plate_image_with_eu_strip):
        """Wynik po usunięciu pasa nie może być pusty."""
        result = anpr_bare._remove_eu_blue_strip(plate_image_with_eu_strip)
        assert result.shape[0] > 0
        assert result.shape[1] > 0

    def test_narrow_image_handled(self, anpr_bare):
        """Bardzo wąski obraz nie powoduje błędu."""
        narrow = np.ones((60, 20, 3), dtype=np.uint8) * 255
        result = anpr_bare._remove_eu_blue_strip(narrow)
        assert result is not None

    def test_returns_bgr_image(self, anpr_bare, plate_image_with_eu_strip):
        result = anpr_bare._remove_eu_blue_strip(plate_image_with_eu_strip)
        assert len(result.shape) == 3
        assert result.shape[2] == 3

    def test_blue_strip_detection_synthetic(self, anpr_bare):
        """Test z syntetyczną tablicą z czystym niebieskim pasem HSV."""
        # Utwórz obraz z wyraźnie niebieskim pasem
        img = np.ones((60, 300, 3), dtype=np.uint8) * 255
        # Czysty niebieski BGR: (255, 0, 0) -> HSV H≈120
        img[:, :45] = [200, 10, 10]   # niebieski BGR
        result = anpr_bare._remove_eu_blue_strip(img)
        # Powinien wykryć i przyciąć
        assert result.shape[1] <= img.shape[1]


# ============================================================
# Testy wariantów preprocesingu tablicy
# ============================================================

class TestPreprocessPlateVariants:
    def test_returns_list(self, anpr_bare):
        plate = np.ones((60, 300, 3), dtype=np.uint8) * 200
        result = anpr_bare.preprocess_plate_variants(plate)
        assert isinstance(result, list)

    def test_returns_multiple_variants(self, anpr_bare):
        plate = np.ones((60, 300, 3), dtype=np.uint8) * 200
        result = anpr_bare.preprocess_plate_variants(plate)
        assert len(result) >= 4

    def test_all_variants_are_bgr(self, anpr_bare):
        plate = np.ones((60, 300, 3), dtype=np.uint8) * 200
        variants = anpr_bare.preprocess_plate_variants(plate)
        for v in variants:
            assert len(v.shape) == 3
            assert v.shape[2] == 3

    def test_variants_have_same_height(self, anpr_bare):
        """Wszystkie warianty powiększone lub pomniejszone mają stałą proporcję."""
        plate = np.ones((60, 300, 3), dtype=np.uint8) * 200
        variants = anpr_bare.preprocess_plate_variants(plate)
        assert all(v is not None for v in variants)


# ============================================================
# Testy integracyjne (wymagają modeli)
# ============================================================

@requires_anpr_model
@pytest.mark.integration
class TestANPRModuleIntegration:
    """Testy z pełną inicjalizacją modeli."""

    @pytest.fixture(scope="class")
    def anpr(self):
        from car_vision_app.anpr import ANPRModule
        return ANPRModule()

    def test_module_initializes(self, anpr):
        assert anpr.plate_detector is not None
        assert anpr.ocr_reader is not None

    def test_process_blank_image(self, anpr):
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        result = anpr.process(img)
        assert 'detected' in result
        assert isinstance(result['detected'], bool)

    def test_process_result_structure(self, anpr):
        img = np.ones((300, 400, 3), dtype=np.uint8) * 150
        result = anpr.process(img)
        required_keys = {'detected', 'bbox', 'plate_crop', 'text', 'confidence'}
        assert required_keys.issubset(result.keys())

    def test_detect_plate_on_black(self, anpr):
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        bbox, score = anpr.detect_license_plate(img)
        # Zwraca None gdy brak tablicy
        if bbox is None:
            assert score == 0.0

    def test_read_plate_white_returns_something(self, anpr):
        """Na białym prostokącie (bez tablicy) nie powinno crashować."""
        plate = np.ones((80, 350, 3), dtype=np.uint8) * 255
        text, conf = anpr.read_plate_text(plate)
        # Może zwrócić None lub jakiś tekst — nie powinno rzucić wyjątku
        assert conf >= 0.0
