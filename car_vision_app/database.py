"""
Moduł zarządzania bazą danych dla przechowywania wyników analizy.
Przechowuje odkryte marki pojazdów i tablice rejestracyjne wraz z obrazkami.
"""

import sqlite3
import os
import hashlib
import shutil
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
            self.conn = sqlite3.connect(str(self.db_path))
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
                    image_hash TEXT UNIQUE,
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
                    
                    -- Metadata
                    notes TEXT
                )
            ''')
            
            # Indeksy dla szybszych zapytań
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_plate_text ON detections(plate_text)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_car_brand ON detections(car_brand)')
            
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
            
            # Wstaw do bazy
            cursor = self.conn.cursor()
            
            cursor.execute('''
                INSERT INTO detections (
                    image_hash, image_filename,
                    car_detected, car_image_filename, car_brand, brand_confidence,
                    plate_detected, plate_image_filename, plate_text, plate_confidence,
                    notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                notes
            ))
            
            self.conn.commit()
            detection_id = cursor.lastrowid
            logger.info(f"✓ Detection added (ID: {detection_id})")
            return detection_id
            
        except sqlite3.IntegrityError:
            logger.warning(f"Duplicate image hash: {image_hash}")
            raise
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
    
    def export_to_csv(self, filepath: str, plate_text: Optional[str] = None):
        """
        Eksportuj dane do pliku CSV (bez obrazków, tylko metadane).
        
        Args:
            filepath: Ścieżka do pliku CSV
            plate_text: Filtruj po tablicy (opcjonalne)
        """
        import csv
        
        try:
            if plate_text:
                detections = self.get_detections_by_plate(plate_text)
            else:
                detections = self.get_all_detections(limit=10000)
            
            if not detections:
                logger.warning("No detections to export")
                return
            
            keys = detections[0].keys()
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=keys)
                writer.writeheader()
                writer.writerows(detections)
            
            logger.info(f"✓ Data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            raise
    
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
