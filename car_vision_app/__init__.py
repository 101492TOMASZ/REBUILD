"""
Pakiet car_vision_app - Aplikacja do rozpoznawania pojazdów.
"""

from .detection import CarDetector
from .classification import BrandClassifier
from .anpr import ANPRModule

__version__ = "1.0.0"
__all__ = ['CarDetector', 'BrandClassifier', 'ANPRModule']
