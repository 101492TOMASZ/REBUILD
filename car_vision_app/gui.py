"""
Nowoczesna aplikacja GUI do analizy pojazdów.
Wykorzystuje PySide6 z ciemnym motywem i płaskim designem.
"""

import sys
import os
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional

# Ustawienie dla PyTorch 2.6+
import torch
torch.serialization.add_safe_globals([])

logger = logging.getLogger(__name__)

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QFrame, QDialog,
    QMessageBox, QGraphicsDropShadowEffect, QSizePolicy, QSpacerItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox, QLineEdit
)
from PySide6.QtCore import Qt, QThread, Signal, QSize, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPixmap, QImage, QFont, QColor

try:
    from .detection import CarDetector
    from .classification import BrandClassifier
    from .anpr import ANPRModule
    from .database import Database
except ImportError:
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from detection import CarDetector
    from classification import BrandClassifier
    from anpr import ANPRModule
    from database import Database


# ==================== STYLE SHEET ====================
# Layer 0 (bg):      #111827  — najciemniejsze, tło okna
# Layer 1 (card):    #1f2937  — karty, wyraźnie jaśniejsze
# Layer 2 (inset):   #111827  — zagłębienia wewnątrz kart (powrót do bg)
DARK_STYLE = """
/* Main Window */
QMainWindow {
    background-color: #111827;
}

QWidget {
    background-color: #111827;
    color: #e2e8f0;
    font-family: 'Segoe UI', 'SF Pro Display', -apple-system, sans-serif;
}

/* Cards */
QFrame[class="card"] {
    background-color: #1f2937;
    border: none;
    border-radius: 16px;
    padding: 16px;
}

/* Labels */
QLabel {
    color: #e2e8f0;
    background-color: transparent;
}

QLabel[class="title"] {
    font-size: 28px;
    font-weight: 700;
    color: #f9fafb;
    background-color: transparent;
    letter-spacing: -0.5px;
}

QLabel[class="subtitle"] {
    font-size: 14px;
    color: #9ca3af;
    background-color: transparent;
}

QLabel[class="section-title"] {
    font-size: 13px;
    font-weight: 700;
    color: #9ca3af;
    background-color: transparent;
    padding: 8px 0px;
    letter-spacing: 1px;
}

QLabel[class="result-value"] {
    font-size: 34px;
    font-weight: 800;
    color: #38bdf8;
    background-color: transparent;
    padding: 4px;
    border-radius: 8px;
    letter-spacing: -1px;
}

QLabel[class="result-label"] {
    font-size: 10px;
    color: #6b7280;
    background-color: transparent;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
}

/* Buttons */
QPushButton {
    border: none;
    border-radius: 10px;
    padding: 12px 24px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.3px;
    min-height: 20px;
}

QPushButton[class="primary"] {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #059669, stop:1 #10b981);
    color: white;
}

QPushButton[class="primary"]:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #10b981, stop:1 #34d399);
}

QPushButton[class="primary"]:pressed {
    background: #047857;
}

QPushButton[class="primary"]:disabled {
    background: #374151;
    color: #6b7280;
}

QPushButton[class="secondary"] {
    background: #374151;
    color: #d1d5db;
    border: none;
}

QPushButton[class="secondary"]:hover {
    background: #4b5563;
    color: #f3f4f6;
}

QPushButton[class="secondary"]:pressed {
    background: #1f2937;
}

QPushButton[class="accent"] {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #0284c7, stop:1 #06b6d4);
    color: white;
}

QPushButton[class="accent"]:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #06b6d4, stop:1 #22d3ee);
}

QPushButton[class="accent"]:disabled {
    background: #374151;
    color: #6b7280;
}

QPushButton[class="danger"] {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #dc2626, stop:1 #ef4444);
    color: white;
}

QPushButton[class="danger"]:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #ef4444, stop:1 #f87171);
}

/* Scrollbar */
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    border-radius: 4px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #4b5563;
    border-radius: 4px;
    min-height: 40px;
    border: none;
}

QScrollBar::handle:vertical:hover {
    background: #6b7280;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    border: none;
    background: none;
}

/* Message Box */
QMessageBox {
    background-color: #1f2937;
    border: none;
    border-radius: 12px;
}

QMessageBox QLabel {
    color: #e2e8f0;
    background-color: transparent;
}

/* Dialog */
QDialog {
    background-color: #111827;
    border-radius: 12px;
}
"""


def cv2_to_qpixmap(cv_img: np.ndarray, max_width: int = None, max_height: int = None) -> QPixmap:
    """Konwertuje obraz OpenCV (BGR) na QPixmap."""
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_img.shape
    bytes_per_line = ch * w
    q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(q_img)
    
    if max_width or max_height:
        pixmap = pixmap.scaled(
            max_width or pixmap.width(),
            max_height or pixmap.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
    return pixmap


class AnalysisWorker(QThread):
    """Wątek roboczy do analizy obrazu - detekcja samochodu i marki."""
    
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(str, int)
    
    def __init__(self, image: np.ndarray, car_detector: CarDetector,
                 brand_classifier: BrandClassifier):
        super().__init__()
        self.image = image
        self.car_detector = car_detector
        self.brand_classifier = brand_classifier
    
    def run(self):
        try:
            results = {}
            
            self.progress.emit("Wykrywanie pojazdu...", 33)
            car_crop, car_data, annotated_image = self.car_detector.detect_and_crop(self.image)
            
            results['annotated_image'] = annotated_image
            results['car_detected'] = car_crop is not None
            
            if car_crop is None:
                results['brand'] = "Brak"
                results['brand_confidence'] = 0.0
                results['car_crop'] = None
                self.finished.emit(results)
                return
            
            results['car_crop'] = car_crop
            
            self.progress.emit("Rozpoznawanie marki...", 66)
            brand, confidence = self.brand_classifier.predict(car_crop)
            results['brand'] = brand
            results['brand_confidence'] = confidence
            
            self.progress.emit("Zakończono!", 100)
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))


class ANPRWorker(QThread):
    """Osobny wątek do odczytu tablicy rejestracyjnej."""
    
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(str)
    
    def __init__(self, car_crop: np.ndarray, anpr_module: ANPRModule):
        super().__init__()
        self.car_crop = car_crop
        self.anpr_module = anpr_module
    
    def run(self):
        try:
            self.progress.emit("Odczyt tablicy...")
            anpr_result = self.anpr_module.process(self.car_crop)
            
            results = {
                'plate_detected': anpr_result['detected'],
                'plate_text': anpr_result['text'],
                'plate_confidence': anpr_result['confidence'],
                'plate_crop': anpr_result['plate_crop'],
                'bbox': anpr_result.get('bbox')
            }
            
            if anpr_result['detected'] and anpr_result['bbox']:
                results['car_crop_with_plate'] = self.anpr_module.draw_plate_bbox(
                    self.car_crop, anpr_result['bbox'], anpr_result['text']
                )
            
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))


class ScaledLabel(QLabel):
    """QLabel ktory automatycznie skaluje pixmap do swojego rozmiaru zachowujac proporcje."""

    def __init__(self):
        super().__init__()
        self._source_pixmap = None
        self.setAlignment(Qt.AlignCenter)

    def setPixmap(self, pixmap: QPixmap):
        self._source_pixmap = pixmap
        self._apply_scaled()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_scaled()

    def _apply_scaled(self):
        if self._source_pixmap and not self._source_pixmap.isNull():
            scaled = self._source_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            super().setPixmap(scaled)

    def clear(self):
        self._source_pixmap = None
        super().clear()


class ImageCard(QFrame):
    """Nowoczesna karta do wyświetlania obrazu."""

    clicked = Signal()

    def __init__(self, title: str = "", placeholder: str = ""):
        super().__init__()
        self._clickable = False
        self.setProperty("class", "card")
        self.setMinimumSize(280, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        if title:
            self.title_label = QLabel(title)
            self.title_label.setProperty("class", "section-title")
            layout.addWidget(self.title_label)
        
        self.image_container = QFrame()
        self.image_container.setStyleSheet("""
            QFrame {
                background-color: #111827;
                border: none;
                border-radius: 12px;
            }
        """)
        self.image_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        img_layout = QVBoxLayout(self.image_container)
        img_layout.setContentsMargins(4, 4, 4, 4)
        img_layout.setAlignment(Qt.AlignCenter)
        
        self.image_label = ScaledLabel()
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.placeholder_text = placeholder
        
        self.placeholder_label = QLabel(placeholder)
        self.placeholder_label.setStyleSheet("color: #64748b; font-size: 13px; font-weight: 500;")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        img_layout.addWidget(self.image_label)
        img_layout.addWidget(self.placeholder_label)
        
        layout.addWidget(self.image_container, 1)
        
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(24)
        shadow.setXOffset(0)
        shadow.setYOffset(8)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(shadow)

    def set_clickable(self, enabled: bool):
        self._clickable = enabled
        self.setCursor(Qt.PointingHandCursor if enabled else Qt.ArrowCursor)

    def enterEvent(self, event):
        if self._clickable and self.isEnabled():
            self.setStyleSheet("""
                QFrame {
                    background-color: #253347;
                    border: none;
                    border-radius: 16px;
                }
            """)
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self._clickable:
            self.setStyleSheet("")
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if self._clickable and self.isEnabled():
            self.clicked.emit()
        super().mousePressEvent(event)

    def set_image(self, pixmap: QPixmap):
        self.image_label.setPixmap(pixmap)
        self.placeholder_label.hide()
    
    def clear_image(self):
        self.image_label.clear()
        self.placeholder_label.show()


class ResultCard(QFrame):
    """Karta wyświetlająca pojedynczy wynik."""
    
    def __init__(self, icon: str, label: str, accent_color: str = "#06b6d4"):
        super().__init__()
        self.setProperty("class", "card")
        self.accent_color = accent_color
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(10)
        
        header = QHBoxLayout()
        
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 28px;")
        header.addWidget(icon_label)
        
        self.label = QLabel(label)
        self.label.setProperty("class", "result-label")
        header.addWidget(self.label)
        header.addStretch()
        
        layout.addLayout(header)
        
        self.value_label = QLabel("---")
        self.value_label.setProperty("class", "result-value")
        self.value_label.setStyleSheet(f"color: {accent_color};")
        layout.addWidget(self.value_label)
        
        self.subtitle_label = QLabel("")
        self.subtitle_label.setProperty("class", "subtitle")
        layout.addWidget(self.subtitle_label)
        
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(24)
        shadow.setXOffset(0)
        shadow.setYOffset(8)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(shadow)
    
    def set_value(self, value: str, subtitle: str = ""):
        self.value_label.setText(value)
        self.subtitle_label.setText(subtitle)
    
    def reset(self):
        self.value_label.setText("---")
        self.subtitle_label.setText("")


class PlateCard(QFrame):
    """Specjalna karta dla tablicy rejestracyjnej."""
    
    def __init__(self):
        super().__init__()
        self.setProperty("class", "card")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        
        header = QLabel("🔢  TABLICA REJESTRACYJNA")
        header.setProperty("class", "result-label")
        layout.addWidget(header)
        
        self.plate_frame = QFrame()
        self.plate_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff, stop:0.5 #f8f8f8, stop:1 #eeeeee);
                border: none;
                border-radius: 6px;
                min-height: 60px;
            }
        """)
        
        plate_layout = QHBoxLayout(self.plate_frame)
        plate_layout.setContentsMargins(8, 8, 8, 8)
        
        
        self.plate_text = QLabel("---")
        self.plate_text.setStyleSheet("""
            font-family: 'FE-Schrift', 'Consolas', 'Courier New', monospace;
            font-size: 36px;
            font-weight: bold;
            color: #1a1a2e;
            letter-spacing: 4px;
            padding: 0 16px;
        """)
        self.plate_text.setAlignment(Qt.AlignCenter)
        plate_layout.addWidget(self.plate_text, 1)
        
        layout.addWidget(self.plate_frame)
        
        self.confidence_label = QLabel("")
        self.confidence_label.setProperty("class", "subtitle")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.confidence_label)
        
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(25)
        shadow.setXOffset(0)
        shadow.setYOffset(6)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(shadow)
    
    def set_plate(self, text: str, confidence: float = 0.0):
        self.plate_text.setText(text if text else "---")
        if text and text != "---" and text != "Nie wykryto tablicy":
            self.confidence_label.setText(f"Pewność: {confidence * 100:.1f}%")
        else:
            self.confidence_label.setText("")
    
    def reset(self):
        self.plate_text.setText("---")
        self.confidence_label.setText("")


class HamburgerButton(QPushButton):
    """Przycisk z trzema kreskami (hamburger menu)."""

    def paintEvent(self, event):
        super().paintEvent(event)
        from PySide6.QtGui import QPainter, QPen
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("#e2e8f0"), 2.5)
        p.setPen(pen)
        w, h = self.width(), self.height()
        cx = w // 2
        for dy in (-6, 0, 6):
            y = h // 2 + dy
            p.drawLine(cx - 9, y, cx + 9, y)
        p.end()


class StatusBar(QFrame):
    """Pasek statusu."""
    
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QFrame {
                background-color: #1f2937;
                border: none;
                border-radius: 10px;
            }
        """)
        self.setFixedHeight(50)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        
        self.status_icon = QLabel("⚡")
        self.status_icon.setStyleSheet("font-size: 18px;")
        layout.addWidget(self.status_icon)
        
        self.status_text = QLabel("Gotowy do pracy")
        self.status_text.setStyleSheet("color: #8b949e; font-size: 13px;")
        layout.addWidget(self.status_text)
        
        layout.addStretch()
        
        self.progress_text = QLabel("")
        self.progress_text.setStyleSheet("color: #58a6ff; font-size: 13px; font-weight: 600;")
        layout.addWidget(self.progress_text)
    
    def set_status(self, text: str, icon: str = "⚡", progress: int = None):
        self.status_icon.setText(icon)
        self.status_text.setText(text)
        if progress is not None:
            self.progress_text.setText(f"{progress}%")
        else:
            self.progress_text.setText("")


class CropsWindow(QDialog):
    """Niemodalny panel z cropami detekcji, wyćcia i tablicy."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Podgląd crops")
        self.setMinimumSize(800, 340)
        self.setStyleSheet(DARK_STYLE)
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        self.detection_card = ImageCard("DETEKCJA", "Wykryty pojazd")
        layout.addWidget(self.detection_card)

        self.crop_card = ImageCard("WYCIĘCIE", "Crop do analizy")
        layout.addWidget(self.crop_card)

        self.plate_crop_card = ImageCard("TABLICA", "Crop tablicy")
        layout.addWidget(self.plate_crop_card)

    def clear_all(self):
        self.detection_card.clear_image()
        self.crop_card.clear_image()
        self.plate_crop_card.clear_image()


class MainWindow(QMainWindow):
    """Główne okno aplikacji."""
    
    def __init__(self, model_path: str, classes_path: str, plate_model_path: str = None):
        super().__init__()
        
        self.model_path = model_path
        self.classes_path = classes_path
        self.plate_model_path = plate_model_path
        
        self.current_image = None
        self.car_detector = None
        self.brand_classifier = None
        self.anpr_module = None
        self.worker = None
        self.anpr_worker = None
        
        # Inicjalizacja bazy danych
        try:
            self.db = Database()
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self.db = None
        
        self.last_car_crop = None
        self.last_brand = None
        self.crops_window = None
        self._last_annotated = None
        self._last_car_crop_img = None
        self._last_plate_crop = None
        self._last_car_crop_with_plate = None
        
        self.init_ui()
        self.load_models()
    
    def init_ui(self):
        """Inicjalizacja interfejsu użytkownika."""
        self.setWindowTitle("CarVision AI")
        self.setMinimumSize(900, 600)

        central = QWidget()
        self.setCentralWidget(central)

        # ===== MAIN AREA (cały central) =====
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(16)

        # ===== TOP BAR =====
        self.top_bar_widget = QWidget()
        self.top_bar_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        top_bar = QHBoxLayout(self.top_bar_widget)
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(14)

        self.hamburger_btn = HamburgerButton()
        self.hamburger_btn.setFixedSize(44, 44)
        self.hamburger_btn.setStyleSheet("""
            QPushButton {
                background-color: #374151;
                border: none;
                border-radius: 10px;
            }
            QPushButton:hover { background-color: #4b5563; }
            QPushButton:pressed { background-color: #1f2937; }
        """)
        self.hamburger_btn.setCursor(Qt.PointingHandCursor)
        self.hamburger_btn.clicked.connect(self.toggle_sidebar)
        top_bar.addWidget(self.hamburger_btn)

        title_lbl = QLabel("🚗 CarVision AI")
        title_lbl.setProperty("class", "title")
        top_bar.addWidget(title_lbl)

        subtitle_lbl = QLabel("Rozpoznawanie marki i tablicy rejestracyjnej")
        subtitle_lbl.setProperty("class", "subtitle")
        subtitle_lbl.setAlignment(Qt.AlignVCenter)
        top_bar.addWidget(subtitle_lbl)

        top_bar.addStretch()
        main_layout.addWidget(self.top_bar_widget)

        # ===== CONTENT =====
        content = QHBoxLayout()
        content.setSpacing(20)

        self.input_card = ImageCard("OBRAZ WEJŚCIOWY", "Kliknij, aby wczytać zdjęcie")
        self.input_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.input_card.set_clickable(True)
        self.input_card.clicked.connect(self.load_image)
        content.addWidget(self.input_card, 3)

        # RIGHT: Results
        right_panel = QVBoxLayout()
        right_panel.setSpacing(16)

        self.brand_card = ResultCard("🏎️", "MARKA POJAZDU", "#58a6ff")
        self.brand_card.setMinimumHeight(140)
        self.brand_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_panel.addWidget(self.brand_card, 1)

        self.plate_card = PlateCard()
        self.plate_card.setMinimumHeight(160)
        self.plate_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_panel.addWidget(self.plate_card, 1)

        content.addLayout(right_panel, 1)
        main_layout.addLayout(content, 1)

        # ===== STATUS BAR =====
        self.status_bar = StatusBar()
        main_layout.addWidget(self.status_bar)

        # ===== SIDEBAR OVERLAY (nad contentem, pozycjonowany absolutnie) =====
        self.sidebar = QFrame(central)
        self.sidebar.setFixedWidth(220)
        self.sidebar.setVisible(False)
        self.sidebar.setStyleSheet("""
            QFrame {
                background-color: #1a2333;
                border-top-right-radius: 20px;
                border-bottom-right-radius: 20px;
            }
        """)

        sidebar_shadow = QGraphicsDropShadowEffect()
        sidebar_shadow.setBlurRadius(32)
        sidebar_shadow.setXOffset(8)
        sidebar_shadow.setYOffset(0)
        sidebar_shadow.setColor(QColor(0, 0, 0, 120))
        self.sidebar.setGraphicsEffect(sidebar_shadow)
        self.sidebar.raise_()

        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(16, 24, 16, 24)
        sidebar_layout.setSpacing(10)

        sidebar_title = QLabel("Menu")
        sidebar_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #6b7280; letter-spacing: 1px;")
        sidebar_layout.addWidget(sidebar_title)

        self.heatmap_btn = QPushButton("🔥  Mapa ciepła")
        self.heatmap_btn.setProperty("class", "accent")
        self.heatmap_btn.setMinimumHeight(44)
        self.heatmap_btn.setCursor(Qt.PointingHandCursor)
        self.heatmap_btn.setEnabled(False)
        self.heatmap_btn.clicked.connect(self.show_heatmap)
        sidebar_layout.addWidget(self.heatmap_btn)

        self.crops_btn = QPushButton("🖼️  Cropy")
        self.crops_btn.setProperty("class", "secondary")
        self.crops_btn.setMinimumHeight(44)
        self.crops_btn.setCursor(Qt.PointingHandCursor)
        self.crops_btn.clicked.connect(self.toggle_crops_window)
        sidebar_layout.addWidget(self.crops_btn)

        sidebar_layout.addSpacing(16)
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #374151;")
        sidebar_layout.addWidget(sep)
        sidebar_layout.addSpacing(8)

        self.save_db_btn = QPushButton("💾  Zapisz do bazy")
        self.save_db_btn.setProperty("class", "secondary")
        self.save_db_btn.setMinimumHeight(44)
        self.save_db_btn.setCursor(Qt.PointingHandCursor)
        self.save_db_btn.setEnabled(False)
        self.save_db_btn.clicked.connect(self.save_to_database)
        sidebar_layout.addWidget(self.save_db_btn)

        self.history_btn = QPushButton("📋  Historia")
        self.history_btn.setProperty("class", "secondary")
        self.history_btn.setMinimumHeight(44)
        self.history_btn.setCursor(Qt.PointingHandCursor)
        self.history_btn.clicked.connect(self.show_history)
        sidebar_layout.addWidget(self.history_btn)

        sidebar_layout.addStretch()

    def _sidebar_y(self):
        """Y od którego zaczyna się sidebar (poniżej top bar)."""
        return self.top_bar_widget.geometry().bottom() + 1

    def _sidebar_target_height(self):
        """Wysokość sidebara = do końca ostatniego przycisku + margines."""
        self.sidebar.adjustSize()
        return self.sidebar.sizeHint().height() + 16

    def resizeEvent(self, event):
        """Aktualizuje pozycję sidebara przy zmianie rozmiaru okna."""
        super().resizeEvent(event)
        if hasattr(self, 'sidebar') and hasattr(self, 'top_bar_widget'):
            y = self._sidebar_y()
            self.sidebar.move(0, y)
            if self.sidebar.isVisible():
                self.sidebar.setFixedHeight(self._sidebar_target_height())

    def toggle_sidebar(self):
        """Wysuwa / chowa panel boczny animacją z góry w dół."""
        y = self._sidebar_y()
        self.sidebar.move(0, y)
        self.sidebar.raise_()

        if not self.sidebar.isVisible():
            target_h = self._sidebar_target_height()
            self.sidebar.setFixedHeight(0)
            self.sidebar.setVisible(True)
            anim = QPropertyAnimation(self.sidebar, b"maximumHeight", self)
            anim.setStartValue(0)
            anim.setEndValue(target_h)
            anim.setDuration(220)
            anim.setEasingCurve(QEasingCurve.OutCubic)
            anim.finished.connect(lambda: self.sidebar.setFixedHeight(target_h))
            self._sidebar_anim = anim
            anim.start()
        else:
            target_h = self.sidebar.height()
            anim = QPropertyAnimation(self.sidebar, b"maximumHeight", self)
            anim.setStartValue(target_h)
            anim.setEndValue(0)
            anim.setDuration(180)
            anim.setEasingCurve(QEasingCurve.InCubic)
            anim.finished.connect(lambda: self.sidebar.setVisible(False))
            self._sidebar_anim = anim
            anim.start()
    
    def load_models(self):
        """Ładuje modele do pamięci."""
        self.status_bar.set_status("Ładowanie modeli...", "⏳")
        QApplication.processEvents()
        
        try:
            self.car_detector = CarDetector('yolov8s.pt')
            self.status_bar.set_status("Ładowanie klasyfikatora...", "⏳", 33)
            QApplication.processEvents()
            
            self.brand_classifier = BrandClassifier(self.model_path, self.classes_path)
            self.status_bar.set_status("Ładowanie modułu ANPR...", "⏳", 66)
            QApplication.processEvents()
            
            self.anpr_module = ANPRModule(self.plate_model_path)
            self.status_bar.set_status("Wszystkie modele załadowane", "✅")
            
        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Nie udało się załadować modeli:\n{str(e)}")
            self.status_bar.set_status(f"Błąd: {str(e)}", "❌")
    
    def load_image(self):
        """Wczytuje obraz z dysku."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Wybierz zdjęcie", "",
            "Obrazy (*.png *.jpg *.jpeg *.bmp *.webp);;Wszystkie (*.*)"
        )
        
        if file_path:
            self.current_image = cv2.imread(file_path)
            
            if self.current_image is None:
                QMessageBox.warning(self, "Błąd", "Nie udało się wczytać obrazu.")
                return
            
            pixmap = cv2_to_qpixmap(self.current_image)
            self.input_card.set_image(pixmap)
            
            self.clear_results()
            self.status_bar.set_status(f"Wczytano: {os.path.basename(file_path)}", "📷")
            self.analyze_image()
    
    def toggle_crops_window(self):
        """Otwiera lub zamyka okno z cropami."""
        if self.crops_window is None or not self.crops_window.isVisible():
            if self.crops_window is None:
                self.crops_window = CropsWindow(self)
            # Uzupełnij okno ostatnimi wynikami jeśli istnieją
            if self._last_annotated is not None:
                pixmap = cv2_to_qpixmap(self._last_annotated, max_width=320, max_height=240)
                self.crops_window.detection_card.set_image(pixmap)
            if self._last_car_crop_with_plate is not None:
                pixmap = cv2_to_qpixmap(self._last_car_crop_with_plate, max_width=320, max_height=240)
                self.crops_window.crop_card.set_image(pixmap)
            elif self._last_car_crop_img is not None:
                pixmap = cv2_to_qpixmap(self._last_car_crop_img, max_width=320, max_height=240)
                self.crops_window.crop_card.set_image(pixmap)
            if self._last_plate_crop is not None:
                pixmap = cv2_to_qpixmap(self._last_plate_crop, max_width=320, max_height=100)
                self.crops_window.plate_crop_card.set_image(pixmap)
            self.crops_window.show()
            self.crops_window.raise_()
        else:
            self.crops_window.hide()

    def clear_results(self):
        """Czyści wyniki."""
        if self.crops_window is not None:
            self.crops_window.clear_all()
        self.brand_card.reset()
        self.plate_card.reset()
        self.heatmap_btn.setEnabled(False)
        self.last_car_crop = None
        self.last_brand = None
        self._last_annotated = None
        self._last_car_crop_img = None
        self._last_plate_crop = None
        self._last_car_crop_with_plate = None
    
    def analyze_image(self):
        """Uruchamia analizę."""
        if self.current_image is None:
            return
        
        if not all([self.car_detector, self.brand_classifier, self.anpr_module]):
            QMessageBox.warning(self, "Błąd", "Modele nie są załadowane.")
            return
        
        self.input_card.setEnabled(False)
        
        # Resetuj tablicę
        self.plate_card.reset()
        if self.crops_window is not None:
            self.crops_window.plate_crop_card.clear_image()
        
        self.worker = AnalysisWorker(
            self.current_image,
            self.car_detector,
            self.brand_classifier
        )
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.progress.connect(self.on_progress)
        self.worker.start()
    
    def on_progress(self, message: str, percent: int):
        self.status_bar.set_status(message, "⏳", percent)
    
    def on_analysis_finished(self, results: dict):
        """Obsługuje zakończenie analizy detekcji i marki."""
        self.input_card.setEnabled(True)
        
        self.last_car_crop = results.get('car_crop')
        self.last_brand = results.get('brand')
        
        if self.last_car_crop is not None:
            self.heatmap_btn.setEnabled(True)
        
        # Zapisz i wyświetl detekcję i crop
        if results.get('annotated_image') is not None:
            self._last_annotated = results['annotated_image']
            if self.crops_window is not None:
                pixmap = cv2_to_qpixmap(self._last_annotated, max_width=320, max_height=240)
                self.crops_window.detection_card.set_image(pixmap)
        if results.get('car_crop') is not None:
            self._last_car_crop_img = results['car_crop']
            if self.crops_window is not None:
                pixmap = cv2_to_qpixmap(self._last_car_crop_img, max_width=320, max_height=240)
                self.crops_window.crop_card.set_image(pixmap)
        
        # Wyświetl markę
        brand = results.get('brand', '---')
        confidence = results.get('brand_confidence', 0.0)
        self.brand_card.set_value(brand, f"Pewność: {confidence:.1f}%")
        
        if confidence >= 80:
            color = "#3fb950"
        elif confidence >= 50:
            color = "#d29922"
        else:
            color = "#f85149"
        self.brand_card.value_label.setStyleSheet(f"color: {color}; font-size: 32px; font-weight: 700;")
        
        # Włącz przycisk zapisu do bazy jeśli istnieje baza danych
        if self.db is not None:
            self.save_db_btn.setEnabled(True)
        
        self.status_bar.set_status("Detekcja zakończona", "✅")
        
        # Uruchom osobno odczyt tablicy (nie blokuje UI)
        if self.last_car_crop is not None:
            self.start_anpr()
    
    def start_anpr(self):
        """Uruchamia osobny wątek do odczytu tablicy."""
        self.plate_card.set_plate("...", 0.0)
        self.status_bar.set_status("Odczyt tablicy...", "🔢")
        
        self.anpr_worker = ANPRWorker(self.last_car_crop, self.anpr_module)
        self.anpr_worker.finished.connect(self.on_anpr_finished)
        self.anpr_worker.error.connect(self.on_anpr_error)
        self.anpr_worker.start()
    
    def on_anpr_finished(self, results: dict):
        """Obsługuje zakończenie odczytu tablicy."""
        plate_text = results.get('plate_text', '---')
        plate_conf = results.get('plate_confidence', 0.0)
        self.plate_card.set_plate(plate_text, plate_conf)
        
        if results.get('plate_crop') is not None:
            self._last_plate_crop = results['plate_crop']
            if self.crops_window is not None:
                pixmap = cv2_to_qpixmap(self._last_plate_crop, max_width=320, max_height=100)
                self.crops_window.plate_crop_card.set_image(pixmap)
        if results.get('car_crop_with_plate') is not None:
            self._last_car_crop_with_plate = results['car_crop_with_plate']
            if self.crops_window is not None:
                pixmap = cv2_to_qpixmap(self._last_car_crop_with_plate, max_width=320, max_height=240)
                self.crops_window.crop_card.set_image(pixmap)
        
        self.status_bar.set_status("Analiza zakończona", "✅")
    
    def on_anpr_error(self, error: str):
        self.plate_card.set_plate("Błąd", 0.0)
        self.status_bar.set_status(f"Błąd tablicy: {error}", "⚠️")
    
    def on_analysis_error(self, error: str):
        self.input_card.setEnabled(True)
        QMessageBox.critical(self, "Błąd analizy", error)
        self.status_bar.set_status(f"Błąd: {error}", "❌")
    
    def show_heatmap(self):
        """Wyświetla mapę ciepła."""
        if self.last_car_crop is None or self.brand_classifier is None:
            return
        
        self.status_bar.set_status("Generowanie mapy ciepła...", "⏳")
        QApplication.processEvents()
        
        try:
            overlay, heatmap, cam = self.brand_classifier.generate_gradcam(self.last_car_crop)
            dialog = HeatmapDialog(self, overlay, heatmap, self.last_car_crop, self.last_brand)
            dialog.exec()
            self.status_bar.set_status("Mapa ciepła wygenerowana", "✅")
        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Nie udało się wygenerować mapy ciepła:\n{str(e)}")
            self.status_bar.set_status(f"Błąd: {str(e)}", "❌")
    
    def save_to_database(self):
        """Zapisuje wynik analizy do bazy danych."""
        if self.db is None:
            QMessageBox.warning(self, "Baza niedostępna", "Baza danych nie jest dostępna")
            return
        
        if self.current_image is None:
            QMessageBox.warning(self, "Brak obrazu", "Wczytaj obraz por przed zapisem")
            return
        
        try:
            self.status_bar.set_status("Zapisywanie do bazy danych...", "⏳")
            QApplication.processEvents()
            
            # Pobierz rozpoznane wyniki
            brand_text = self.brand_card.value_label.text()
            car_detected = brand_text != "---"
            
            plate_text = self.plate_card.plate_text.text()
            plate_detected = plate_text and plate_text != "---" and plate_text != "Nie wykryto tablicy"
            
            # Pobierz confidence values
            brand_confidence = 0.0
            plate_confidence = 0.0
            try:
                if self.brand_card.subtitle_label.text():
                    brand_confidence = float(self.brand_card.subtitle_label.text().split(": ")[1].rstrip("%")) / 100
                if self.plate_card.confidence_label.text():
                    plate_confidence = float(self.plate_card.confidence_label.text().split(": ")[1].rstrip("%")) / 100
            except:
                pass
            
            # Zapisz do bazy
            detection_id = self.db.add_detection(
                image=self.current_image,
                car_detected=car_detected,
                car_image=self.last_car_crop if self.last_car_crop is not None else None,
                car_brand=brand_text if car_detected else None,
                brand_confidence=brand_confidence,
                plate_detected=plate_detected,
                plate_image=None,  # TODO: przesłać plate_crop jeśli dostępny
                plate_text=plate_text if plate_detected else None,
                plate_confidence=plate_confidence,
                notes=None
            )
            
            self.status_bar.set_status(f"✅ Zapisano do bazy (ID: {detection_id})", "✅")
            QMessageBox.information(self, "Sukces", f"Wynik został zapisany do bazy danych.\nID: {detection_id}\n\nLokalizacja: {self.db.db_dir}")
            
            # Wyłącz przycisk po zapisie
            self.save_db_btn.setEnabled(False)
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            QMessageBox.critical(self, "Błąd", f"Nie udało się zapisać do bazy:\n{str(e)}")
            self.status_bar.set_status(f"Błąd zapisu: {str(e)}", "❌")
    
    def closeEvent(self, event):
        """Obsługuje zamknięcie aplikacji - zamyka bazę danych."""
        if self.db is not None:
            try:
                self.db.close()
                logger.info("Database closed successfully")
            except Exception as e:
                logger.error(f"Error closing database: {e}")
        event.accept()
    
    def show_history(self):
        """Wyświetla okno historii z bazą danych."""
        if self.db is None:
            QMessageBox.warning(self, "Baza niedostępna", "Baza danych nie jest dostępna")
            return
        
        dialog = HistoryDialog(self, self.db)
        dialog.exec()


class HistoryDialog(QDialog):
    """Okno dialogowe do przeglądania historii analiz z bazą danych."""
    
    def __init__(self, parent, database):
        super().__init__(parent)
        
        self.db = database
        self.setWindowTitle("Historia analiz")
        self.setMinimumSize(1200, 700)
        self.setStyleSheet(DARK_STYLE)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        
        # ===== HEADER =====
        header = QVBoxLayout()
        title = QLabel("📋 Historia analiz")
        title.setStyleSheet("font-size: 24px; font-weight: 700; color: #f0f6fc;")
        header.addWidget(title)
        layout.addLayout(header)
        
        # ===== FILTERS =====
        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(12)
        
        filter_label = QLabel("Filtruj:")
        filter_label.setStyleSheet("color: #8b949e; font-weight: 600;")
        filter_layout.addWidget(filter_label)
        
        self.brand_filter = QLineEdit()
        self.brand_filter.setPlaceholderText("Filtruj po marce...")
        self.brand_filter.setMaximumWidth(200)
        self.brand_filter.setStyleSheet("""
            QLineEdit {
                background-color: #374151;
                border: none;
                border-radius: 8px;
                padding: 8px;
                color: #e2e8f0;
            }
        """)
        self.brand_filter.textChanged.connect(self.refresh_table)
        filter_layout.addWidget(self.brand_filter)
        
        self.plate_filter = QLineEdit()
        self.plate_filter.setPlaceholderText("Filtruj po tablicy...")
        self.plate_filter.setMaximumWidth(200)
        self.plate_filter.setStyleSheet("""
            QLineEdit {
                background-color: #374151;
                border: none;
                border-radius: 8px;
                padding: 8px;
                color: #e2e8f0;
            }
        """)
        self.plate_filter.textChanged.connect(self.refresh_table)
        filter_layout.addWidget(self.plate_filter)
        
        filter_layout.addStretch()
        
        layout.addLayout(filter_layout)
        
        # ===== STATISTICS =====
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(16)
        
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("color: #8b949e; font-size: 12px;")
        stats_layout.addWidget(self.stats_label)
        stats_layout.addStretch()
        
        layout.addLayout(stats_layout)
        
        # ===== TABLE =====
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "ID", "Data/Czas", "Marka", "Pewność %", "Tablica", "Tablica Pewność %", "Auto", "Tablica"
        ])
        
        # Stylowanie tabeli
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #1f2937;
                border: none;
                border-radius: 8px;
                gridline-color: #374151;
            }
            QTableWidget::item {
                padding: 8px;
                color: #d1d5db;
            }
            QTableWidget::item:selected {
                background-color: #065f46;
            }
            QHeaderView::section {
                background-color: #111827;
                color: #f9fafb;
                padding: 8px;
                border: none;
                font-weight: 600;
            }
        """)
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)
        
        layout.addWidget(self.table)
        
        # ===== BUTTONS =====
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        
        refresh_btn = QPushButton("🔄 Odśwież")
        refresh_btn.setProperty("class", "secondary")
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.clicked.connect(self.refresh_table)
        button_layout.addWidget(refresh_btn)
        
        export_btn = QPushButton("📥 Eksportuj CSV")
        export_btn.setProperty("class", "accent")
        export_btn.setCursor(Qt.PointingHandCursor)
        export_btn.clicked.connect(self.export_to_csv)
        button_layout.addWidget(export_btn)
        
        delete_btn = QPushButton("🗑️ Usuń zaznaczone")
        delete_btn.setProperty("class", "danger")
        delete_btn.setCursor(Qt.PointingHandCursor)
        delete_btn.clicked.connect(self.delete_selected)
        button_layout.addWidget(delete_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Zamknij")
        close_btn.setProperty("class", "secondary")
        close_btn.setFixedWidth(120)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        # Załaduj dane
        self.refresh_table()
    
    def refresh_table(self):
        """Odświeża tabelę z danymi z bazy."""
        try:
            self.table.setRowCount(0)
            
            brand_filter = self.brand_filter.text().strip().upper()
            plate_filter = self.plate_filter.text().strip().upper()
            
            # Pobierz wszystkie dane z bazy
            detections = self.db.get_all_detections(limit=1000)
            
            # Filtruj
            filtered = detections
            if brand_filter:
                filtered = [d for d in filtered if d['car_brand'] and brand_filter in d['car_brand'].upper()]
            if plate_filter:
                filtered = [d for d in filtered if d['plate_text'] and plate_filter in d['plate_text'].upper()]
            
            # Aktualizuj statystyki
            total = len(detections)
            with_car = sum(1 for d in detections if d['car_detected'])
            with_plate = sum(1 for d in detections if d['plate_detected'])
            self.stats_label.setText(f"Łącznie: {total} | Z pojazdem: {with_car} | Z tablicą: {with_plate} | Filtrowano: {len(filtered)}")
            
            # Dodaj wiersze do tabeli
            for detection in filtered:
                row = self.table.rowCount()
                self.table.insertRow(row)
                
                # ID
                item_id = QTableWidgetItem(str(detection['id']))
                self.table.setItem(row, 0, item_id)
                
                # Data/Czas
                timestamp = detection['timestamp'][:16] if detection['timestamp'] else ''
                item_time = QTableWidgetItem(timestamp)
                self.table.setItem(row, 1, item_time)
                
                # Marka
                brand = detection['car_brand'] or '---'
                item_brand = QTableWidgetItem(brand)
                self.table.setItem(row, 2, item_brand)
                
                # Pewność marki
                brand_conf = f"{detection['brand_confidence']*100:.1f}%" if detection['brand_confidence'] else '0%'
                item_brand_conf = QTableWidgetItem(brand_conf)
                self.table.setItem(row, 3, item_brand_conf)
                
                # Tablica
                plate = detection['plate_text'] or '---'
                item_plate = QTableWidgetItem(plate)
                self.table.setItem(row, 4, item_plate)
                
                # Pewność tablicy
                plate_conf = f"{detection['plate_confidence']*100:.1f}%" if detection['plate_confidence'] else '0%'
                item_plate_conf = QTableWidgetItem(plate_conf)
                self.table.setItem(row, 5, item_plate_conf)
                
                # Auto (binary indicator)
                car_icon = "✅" if detection['car_detected'] else "❌"
                item_car = QTableWidgetItem(car_icon)
                self.table.setItem(row, 6, item_car)
                
                # Tablica (binary indicator)
                plate_icon = "✅" if detection['plate_detected'] else "❌"
                item_plt = QTableWidgetItem(plate_icon)
                self.table.setItem(row, 7, item_plt)
            
            logger.info(f"Loaded {len(filtered)} detections")
            
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            QMessageBox.critical(self, "Błąd", f"Nie udało się załadować historii:\n{str(e)}")
    
    def export_to_csv(self):
        """Eksportuje dane do pliku CSV."""
        try:
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Eksportuj do CSV", "", "CSV Files (*.csv)"
            )
            
            if not filepath:
                return
            
            self.db.export_to_csv(filepath)
            QMessageBox.information(self, "Sukces", f"Dane zostały wyeksportowane do:\n{filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting: {e}")
            QMessageBox.critical(self, "Błąd", f"Nie udało się wyeksportować:\n{str(e)}")
    
    def delete_selected(self):
        """Usuwa zaznaczone rekordy z bazy."""
        selected_rows = self.table.selectedIndexes()
        if not selected_rows:
            QMessageBox.warning(self, "Brak zaznaczenia", "Zaznacz rekordy do usunięcia")
            return
        
        # Pobierz unikalne wiersze
        rows = set(idx.row() for idx in selected_rows)
        
        reply = QMessageBox.question(
            self, "Potwierdzenie",
            f"Czy na pewno usunąć {len(rows)} rekord(ów)?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        try:
            for row in sorted(rows, reverse=True):
                detection_id = int(self.table.item(row, 0).text())
                self.db.delete_detection(detection_id)
            
            QMessageBox.information(self, "Sukces", f"Usunięto {len(rows)} rekord(ów)")
            self.refresh_table()
            
        except Exception as e:
            logger.error(f"Error deleting: {e}")
            QMessageBox.critical(self, "Błąd", f"Nie udało się usunąć:\n{str(e)}")


class HeatmapDialog(QDialog):
    """Nowoczesne okno dialogowe z mapą ciepła."""
    
    def __init__(self, parent, overlay: np.ndarray, heatmap: np.ndarray, 
                 original: np.ndarray, brand: str):
        super().__init__(parent)
        
        self.setWindowTitle(f"Grad-CAM - {brand}")
        self.setMinimumSize(950, 600)
        self.setStyleSheet(DARK_STYLE)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)
        
        header = QVBoxLayout()
        title = QLabel(f"🔥 Mapa aktywacji dla: {brand}")
        title.setStyleSheet("font-size: 22px; font-weight: 700; color: #f0f6fc;")
        header.addWidget(title)
        
        desc = QLabel("Wizualizacja obszarów obrazu które najbardziej wpłynęły na decyzję modelu")
        desc.setStyleSheet("font-size: 13px; color: #8b949e;")
        header.addWidget(desc)
        layout.addLayout(header)
        
        images = QHBoxLayout()
        images.setSpacing(16)
        
        orig_card = self._create_image_card("Oryginalny obraz", original, "#30363d")
        images.addWidget(orig_card)
        
        heat_card = self._create_image_card("Mapa ciepła", heatmap, "#da3633")
        images.addWidget(heat_card)
        
        over_card = self._create_image_card("Nałożenie", overlay, "#238636")
        images.addWidget(over_card)
        
        layout.addLayout(images)
        
        legend = QLabel("🔴 Czerwony = wysoki wpływ na decyzję  •  🔵 Niebieski = niski wpływ")
        legend.setStyleSheet("color: #8b949e; font-size: 12px;")
        legend.setAlignment(Qt.AlignCenter)
        layout.addWidget(legend)
        
        close_btn = QPushButton("Zamknij")
        close_btn.setProperty("class", "secondary")
        close_btn.setFixedSize(120, 40)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn, alignment=Qt.AlignCenter)
    
    def _create_image_card(self, title: str, image: np.ndarray, border_color: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: #1f2937;
                border: none;
                border-radius: 12px;
            }}
        """)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("font-weight: 600; color: #f0f6fc;")
        lbl_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl_title)
        
        img_label = QLabel()
        pixmap = cv2_to_qpixmap(image, max_width=280, max_height=280)
        img_label.setPixmap(pixmap)
        img_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(img_label)
        
        return card


def run_app(model_path: str, classes_path: str, plate_model_path: str = None):
    """Uruchamia aplikację."""
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = MainWindow(model_path, classes_path, plate_model_path)
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    app_dir = Path(__file__).parent
    model_path = base_dir / "car_detector_model" / "model.pth"
    classes_path = base_dir / "car_detector_model" / "label_map.json"
    # Priorytet: anpr_best.pt w katalogu aplikacji, potem stary model
    plate_model_path = app_dir / "anpr_best.pt"
    if not plate_model_path.exists():
        plate_model_path = base_dir / "Automatic-License-Plate-Recognition-using-YOLOv8-main" / "license_plate_detector.pt"
    run_app(
        str(model_path),
        str(classes_path),
        str(plate_model_path) if plate_model_path.exists() else None,
    )
