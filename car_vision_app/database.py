"""
Moduł zarządzania bazą danych dla przechowywania wyników analizy.
Przechowuje odkryte marki pojazdów i tablice rejestracyjne wraz z obrazkami.
"""

import sqlite3
import os
import hashlib
import shutil
import csv
import io
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Database:
    """Zarządzacz bazą danych dla wyników analizy pojazdów z przechowywaniem obrazków."""
    
    def __init__(self):
        """Inicjalizacja bazy danych w ukrytym folderze użytkownika."""
        # Ukryty folder w home directory
        home_dir = Path.home()
        self.db_dir = home_dir / '.carvision'
        self.db_path = self.db_dir / 'database.db'
        self.images_dir = self.db_dir / 'images'
        
        # Utwórz foldery jeśli nie istnieją
        self.db_dir.mkdir(exist_ok=True, mode=0o700)  # Tylko właściciel ma dostęp
        self.images_dir.mkdir(exist_ok=True, mode=0o700)
        
        logger.info(f"Database path: {self.db_path}")
        logger.info(f"Images directory: {self.images_dir}")
        
        self.conn = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Nawiąż połączenie z bazą danych."""
        try:
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # Dostęp do kolumn po nazwie
            logger.info("✓ Database connection established")
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _create_tables(self):
        """Utwórz tabele jeśli nie istnieją."""
        try:
            cursor = self.conn.cursor()
            
            # Główna tabela z wynikami analizy
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Informacje o obrazie
                    image_hash TEXT,
                    image_filename TEXT,
                    
                    -- Detekcja pojazdu
                    car_detected INTEGER DEFAULT 0,
                    car_image_filename TEXT,
                    
                    -- Rozpoznanie marki
                    car_brand TEXT,
                    brand_confidence REAL,
                    
                    -- Rozpoznanie tablicy
                    plate_detected INTEGER DEFAULT 0,
                    plate_text TEXT,
                    plate_confidence REAL,
                    plate_image_filename TEXT,
                    
                    -- Grad-CAM
                    gradcam_image_filename TEXT,
                    
                    -- Oznaczenie błędnego wyniku
                    is_incorrect INTEGER DEFAULT 0,
                    
                    -- Metadata
                    notes TEXT
                )
            ''')
            
            # Migracja: dodaj kolumny jeśli ich brak (dla istniejących baz)
            for col, col_type, default in [
                ('gradcam_image_filename', 'TEXT', None),
                ('is_incorrect', 'INTEGER', '0'),
            ]:
                try:
                    cursor.execute(f'ALTER TABLE detections ADD COLUMN {col} {col_type} DEFAULT {default}')
                except sqlite3.OperationalError:
                    pass  # kolumna już istnieje
            
            # Indeksy dla szybszych zapytań
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_plate_text ON detections(plate_text)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_car_brand ON detections(car_brand)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_is_incorrect ON detections(is_incorrect)')
            
            # Migracja: usuń UNIQUE constraint z image_hash (jeśli istnieje)
            try:
                cursor.execute("DROP INDEX IF EXISTS sqlite_autoindex_detections_1")
            except sqlite3.OperationalError:
                pass
            
            self.conn.commit()
            logger.info("✓ Database tables created")
            
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    @staticmethod
    def _compute_hash(image: np.ndarray) -> str:
        """Oblicz hash MD5 dla obrazu."""
        if isinstance(image, np.ndarray):
            image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
        else:
            with open(image, 'rb') as f:
                image_bytes = f.read()
        
        return hashlib.md5(image_bytes).hexdigest()
    
    def save_image(self, image: np.ndarray) -> str:
        """
        Zapisz obraz do folderu images/ i zwróć nazwę pliku.
        Jeśli obraz o tym hashu już istnieje, zwróć istniejącą nazwę.
        
        Args:
            image: Obraz OpenCV (numpy array)
        
        Returns:
            Nazwa pliku (np. "abc123def456.jpg")
        """
        image_hash = self._compute_hash(image)
        filename = f"{image_hash}.jpg"
        filepath = self.images_dir / filename
        
        # Jeśli plik już istnieje, zwróć jego nazwę
        if filepath.exists():
            logger.debug(f"Image {filename} already exists")
            return filename
        
        # Zapisz obraz
        try:
            success = cv2.imwrite(str(filepath), image)
            if success:
                logger.info(f"✓ Image saved: {filename}")
                return filename
            else:
                logger.error(f"Failed to write image: {filepath}")
                raise IOError(f"Could not write image to {filepath}")
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            raise
    
    def get_image(self, filename: str) -> Optional[np.ndarray]:
        """
        Pobierz obraz z folderu images/.
        
        Args:
            filename: Nazwa pliku (np. "abc123def456.jpg")
        
        Returns:
            Obraz OpenCV lub None jeśli plik nie istnieje
        """
        filepath = self.images_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Image not found: {filename}")
            return None
        
        try:
            image = cv2.imread(str(filepath))
            return image
        except Exception as e:
            logger.error(f"Error reading image: {e}")
            return None
    
    def add_detection(self, 
                     image: np.ndarray,
                     car_detected: bool = False,
                     car_image: Optional[np.ndarray] = None,
                     car_brand: Optional[str] = None,
                     brand_confidence: float = 0.0,
                     plate_detected: bool = False,
                     plate_image: Optional[np.ndarray] = None,
                     plate_text: Optional[str] = None,
                     plate_confidence: float = 0.0,
                     gradcam_image: Optional[np.ndarray] = None,
                     notes: Optional[str] = None) -> int:
        """
        Dodaj nowy wynik analizy do bazy danych wraz z obrazkami.
        
        Args:
            image: Oryginalny obraz
            car_detected: Czy wykryto pojazd
            car_image: Obraz pojazdu (opcjonalny)
            car_brand: Rozpoznana marka
            brand_confidence: Pewność rozpoznania marki
            plate_detected: Czy wykryto tablicę
            plate_image: Obraz tablicy (opcjonalny)
            plate_text: Tekst tablicy
            plate_confidence: Pewność rozpoznania tablicy
            notes: Notatka
        
        Returns:
            ID dodanego rekordu
        """
        try:
            # Zapisz obrazki
            image_hash = self._compute_hash(image)
            image_filename = self.save_image(image)
            
            car_image_filename = None
            if car_image is not None:
                car_image_filename = self.save_image(car_image)
            
            plate_image_filename = None
            if plate_image is not None:
                plate_image_filename = self.save_image(plate_image)
            
            gradcam_image_filename = None
            if gradcam_image is not None:
                gradcam_image_filename = self.save_image(gradcam_image)
            
            # Wstaw do bazy
            cursor = self.conn.cursor()
            
            cursor.execute('''
                INSERT INTO detections (
                    image_hash, image_filename,
                    car_detected, car_image_filename, car_brand, brand_confidence,
                    plate_detected, plate_image_filename, plate_text, plate_confidence,
                    gradcam_image_filename, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                image_hash,
                image_filename,
                1 if car_detected else 0,
                car_image_filename,
                car_brand,
                brand_confidence,
                1 if plate_detected else 0,
                plate_image_filename,
                plate_text,
                plate_confidence,
                gradcam_image_filename,
                notes
            ))
            
            self.conn.commit()
            detection_id = cursor.lastrowid
            logger.info(f"✓ Detection added (ID: {detection_id})")
            return detection_id

        except Exception as e:
            logger.error(f"Error adding detection: {e}")
            raise
    
    def get_all_detections(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        Pobierz wszystkie rekordy z bazy.
        
        Args:
            limit: Liczba rekordów do pobrania
            offset: Przesunięcie
        
        Returns:
            Lista słowników z wynikami
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                SELECT * FROM detections
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            ''', (limit, offset))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except sqlite3.Error as e:
            logger.error(f"Error fetching detections: {e}")
            return []
    
    def get_detections_by_plate(self, plate_text: str) -> List[Dict]:
        """Pobierz wszystkie rekordy dla danej tablicy rejestracyjnej."""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                SELECT * FROM detections
                WHERE plate_text = ?
                ORDER BY timestamp DESC
            ''', (plate_text,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except sqlite3.Error as e:
            logger.error(f"Error fetching by plate: {e}")
            return []
    
    def get_detections_by_brand(self, brand: str) -> List[Dict]:
        """Pobierz wszystkie rekordy dla danej marki pojazdu."""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                SELECT * FROM detections
                WHERE car_brand = ?
                ORDER BY timestamp DESC
            ''', (brand,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except sqlite3.Error as e:
            logger.error(f"Error fetching by brand: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """Pobierz statystyki z bazy danych."""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('SELECT COUNT(*) as total FROM detections')
            total = cursor.fetchone()['total']
            
            cursor.execute('SELECT COUNT(*) as count FROM detections WHERE car_detected = 1')
            cars_detected = cursor.fetchone()['count']
            
            cursor.execute('SELECT COUNT(*) as count FROM detections WHERE plate_detected = 1')
            plates_detected = cursor.fetchone()['count']
            
            cursor.execute('SELECT COUNT(DISTINCT car_brand) as count FROM detections WHERE car_brand IS NOT NULL')
            unique_brands = cursor.fetchone()['count']
            
            cursor.execute('SELECT COUNT(DISTINCT plate_text) as count FROM detections WHERE plate_text IS NOT NULL')
            unique_plates = cursor.fetchone()['count']
            
            cursor.execute('''
                SELECT car_brand, COUNT(*) as count
                FROM detections
                WHERE car_brand IS NOT NULL
                GROUP BY car_brand
                ORDER BY count DESC
                LIMIT 10
            ''')
            top_brands = [dict(row) for row in cursor.fetchall()]
            
            return {
                'total_detections': total,
                'cars_detected': cars_detected,
                'plates_detected': plates_detected,
                'unique_brands': unique_brands,
                'unique_plates': unique_plates,
                'top_brands': top_brands
            }
            
        except sqlite3.Error as e:
            logger.error(f"Error fetching statistics: {e}")
            return {}
    
    def export_to_csv(self, filepath: str, plate_text: Optional[str] = None, detections: Optional[list] = None):
        """
        Eksportuj dane do pliku CSV (bez obrazków, tylko metadane).
        
        Args:
            filepath: Ścieżka do pliku CSV
            plate_text: Filtruj po tablicy (opcjonalne)
            detections: Lista detekcji do eksportu (None = wszystkie)
        """
        try:
            if detections is None:
                if plate_text:
                    detections = self.get_detections_by_plate(plate_text)
                else:
                    detections = self.get_all_detections(limit=10000)
            
            if not detections:
                logger.warning("No detections to export")
                return
            
            fieldnames = [
                'id', 'timestamp', 'car_brand', 'brand_confidence',
                'plate_text', 'plate_confidence', 'car_detected',
                'plate_detected', 'is_incorrect', 'notes'
            ]
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                for det in detections:
                    row = dict(det)
                    row['brand_confidence'] = f"{(row.get('brand_confidence') or 0)*100:.1f}%"
                    row['plate_confidence'] = f"{(row.get('plate_confidence') or 0)*100:.1f}%"
                    row['car_detected'] = 'Tak' if row.get('car_detected') else 'Nie'
                    row['plate_detected'] = 'Tak' if row.get('plate_detected') else 'Nie'
                    row['is_incorrect'] = 'Tak' if row.get('is_incorrect') else 'Nie'
                    writer.writerow(row)
            
            logger.info(f"✓ Data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            raise
    
    def toggle_incorrect(self, detection_id: int) -> bool:
        """Przełącza flagę is_incorrect dla danego rekordu. Zwraca nową wartość."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                'UPDATE detections SET is_incorrect = 1 - is_incorrect WHERE id = ?',
                (detection_id,)
            )
            self.conn.commit()
            cursor.execute('SELECT is_incorrect FROM detections WHERE id = ?', (detection_id,))
            row = cursor.fetchone()
            new_val = bool(row['is_incorrect']) if row else False
            logger.info(f"Detection {detection_id} is_incorrect toggled to {new_val}")
            return new_val
        except sqlite3.Error as e:
            logger.error(f"Error toggling incorrect flag: {e}")
            return False
    
    def update_gradcam(self, detection_id: int, gradcam_image: np.ndarray) -> Optional[str]:
        """Zapisz obraz Grad-CAM i zaktualizuj rekord w bazie."""
        try:
            filename = self.save_image(gradcam_image)
            cursor = self.conn.cursor()
            cursor.execute(
                'UPDATE detections SET gradcam_image_filename = ? WHERE id = ?',
                (filename, detection_id)
            )
            self.conn.commit()
            logger.info(f"Grad-CAM updated for detection {detection_id}")
            return filename
        except Exception as e:
            logger.error(f"Error updating gradcam: {e}")
            return None
    
    def get_detection_by_id(self, detection_id: int) -> Optional[Dict]:
        """Pobierz pojedynczy rekord po ID."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT * FROM detections WHERE id = ?', (detection_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except sqlite3.Error as e:
            logger.error(f"Error fetching detection {detection_id}: {e}")
            return None
    
    def export_to_pdf(self, filepath: str, detections: Optional[List[Dict]] = None):
        """
        Eksportuj dane do pliku PDF z miniaturami obrazków i Grad-CAM.
        
        Args:
            filepath: Ścieżka do pliku PDF
            detections: Lista detekcji do eksportu (None = wszystkie)
        """
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import mm
            from reportlab.lib import colors
            from reportlab.platypus import (
                SimpleDocTemplate, Table, TableStyle, Paragraph,
                Spacer, Image as RLImage, PageBreak
            )
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
        except ImportError:
            raise ImportError(
                "Brakuje biblioteki reportlab. Zainstaluj: pip install reportlab"
            )
        
        if detections is None:
            detections = self.get_all_detections(limit=10000)
        
        if not detections:
            logger.warning("No detections to export")
            return
        
        doc = SimpleDocTemplate(
            filepath, pagesize=A4,
            leftMargin=15*mm, rightMargin=15*mm,
            topMargin=15*mm, bottomMargin=15*mm
        )
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle', parent=styles['Title'],
            fontSize=22, spaceAfter=6*mm,
            textColor=colors.HexColor('#1a1a2e')
        )
        subtitle_style = ParagraphStyle(
            'CustomSubtitle', parent=styles['Normal'],
            fontSize=10, spaceAfter=8*mm,
            textColor=colors.HexColor('#6b7280')
        )
        section_style = ParagraphStyle(
            'SectionStyle', parent=styles['Heading2'],
            fontSize=13, spaceBefore=4*mm, spaceAfter=2*mm,
            textColor=colors.HexColor('#111827')
        )
        cell_style = ParagraphStyle(
            'CellStyle', parent=styles['Normal'],
            fontSize=8, leading=11
        )
        
        elements = []
        
        # Nagłówek
        elements.append(Paragraph("🚗 CarVision AI — Raport analiz", title_style))
        now = datetime.now().strftime('%Y-%m-%d %H:%M')
        elements.append(Paragraph(f"Wygenerowano: {now}  |  Rekordów: {len(detections)}", subtitle_style))
        
        # Tabela podsumowująca
        total = len(detections)
        with_car = sum(1 for d in detections if d.get('car_detected'))
        with_plate = sum(1 for d in detections if d.get('plate_detected'))
        incorrect = sum(1 for d in detections if d.get('is_incorrect'))
        
        summary_data = [
            ['Łącznie analiz', 'Z pojazdem', 'Z tablicą', 'Oznaczone jako błędne'],
            [str(total), str(with_car), str(with_plate), str(incorrect)],
        ]
        summary_table = Table(summary_data, colWidths=[45*mm]*4)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f2937')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#f3f4f6')),
            ('FONTSIZE', (0, 1), (-1, 1), 14),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1d5db')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 8*mm))
        
        # Szczegóły każdej detekcji
        elements.append(Paragraph("Szczegóły analiz", section_style))
        
        for det in detections:
            det_id = det.get('id', '?')
            timestamp = (det.get('timestamp') or '')[:19]
            brand = det.get('car_brand') or '---'
            brand_conf = f"{(det.get('brand_confidence') or 0)*100:.1f}%"
            plate = det.get('plate_text') or '---'
            plate_conf = f"{(det.get('plate_confidence') or 0)*100:.1f}%"
            is_inc = "⚠ BŁĘDNY" if det.get('is_incorrect') else "✓ OK"
            
            # Wiersz z danymi tekstowymi
            info_data = [
                [Paragraph(f'<b>ID:</b> {det_id}', cell_style),
                 Paragraph(f'<b>Data:</b> {timestamp}', cell_style),
                 Paragraph(f'<b>Status:</b> {is_inc}', cell_style)],
                [Paragraph(f'<b>Marka:</b> {brand} ({brand_conf})', cell_style),
                 Paragraph(f'<b>Tablica:</b> {plate} ({plate_conf})', cell_style),
                 Paragraph(f'<b>Notatki:</b> {det.get("notes") or "—"}', cell_style)],
            ]
            info_table = Table(info_data, colWidths=[60*mm, 60*mm, 60*mm])
            bg_color = colors.HexColor('#fef2f2') if det.get('is_incorrect') else colors.HexColor('#f9fafb')
            info_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), bg_color),
                ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#e5e7eb')),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(info_table)
            
            # Wiersz z obrazkami
            img_cells = []
            thumb_size = 50*mm
            for fname_key, label in [
                ('car_image_filename', 'Pojazd'),
                ('plate_image_filename', 'Tablica'),
                ('gradcam_image_filename', 'Grad-CAM'),
            ]:
                fname = det.get(fname_key)
                if fname:
                    img_path = self.images_dir / fname
                    if img_path.exists():
                        try:
                            img = RLImage(str(img_path), width=thumb_size, height=35*mm)
                            img.hAlign = 'CENTER'
                            img_cells.append([Paragraph(f'<b>{label}</b>', cell_style), img])
                        except Exception:
                            img_cells.append([Paragraph(f'<b>{label}</b>', cell_style),
                                              Paragraph('(błąd obrazu)', cell_style)])
                    else:
                        img_cells.append([Paragraph(f'<b>{label}</b>', cell_style),
                                          Paragraph('(brak pliku)', cell_style)])
                else:
                    img_cells.append([Paragraph(f'<b>{label}</b>', cell_style),
                                      Paragraph('—', cell_style)])
            
            if img_cells:
                # Transpozycja: img_cells to lista [label, img] par — zrob tabele
                labels_row = [c[0] for c in img_cells]
                imgs_row = [c[1] for c in img_cells]
                img_table = Table([labels_row, imgs_row], colWidths=[60*mm]*3)
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f3f4f6')),
                    ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#e5e7eb')),
                    ('TOPPADDING', (0, 0), (-1, -1), 3),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ]))
                elements.append(img_table)
            
            elements.append(Spacer(1, 5*mm))
        
        doc.build(elements)
        logger.info(f"✓ PDF exported to {filepath}")
    
    def delete_detection(self, detection_id: int) -> bool:
        """Usuń rekord z bazy danych (obrazki zostają w folderze)."""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('DELETE FROM detections WHERE id = ?', (detection_id,))
            self.conn.commit()
            
            logger.info(f"✓ Detection {detection_id} deleted")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error deleting detection: {e}")
            return False
    
    def cleanup_unused_images(self) -> int:
        """
        Usuń obrazki które nie są już przypisane do żadnego rekordu.
        
        Returns:
            Liczba usuniętych plików
        """
        try:
            cursor = self.conn.cursor()
            
            # Pobierz wszystkie używane pliki
            cursor.execute('''
                SELECT image_filename FROM detections
                UNION
                SELECT car_image_filename FROM detections
                UNION
                SELECT plate_image_filename FROM detections
                UNION
                SELECT gradcam_image_filename FROM detections
            ''')
            
            used_files = set(row[0] for row in cursor.fetchall() if row[0])
            
            # Usuń pliki które nie są używane
            deleted_count = 0
            for filepath in self.images_dir.glob('*.jpg'):
                if filepath.name not in used_files:
                    try:
                        filepath.unlink()
                        deleted_count += 1
                        logger.info(f"Deleted unused image: {filepath.name}")
                    except Exception as e:
                        logger.error(f"Error deleting {filepath}: {e}")
            
            logger.info(f"✓ Cleaned up {deleted_count} unused images")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up images: {e}")
            return 0
    
    def close(self):
        """Zamknij połączenie z bazą danych."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
