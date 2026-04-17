"""
Pakiet car_vision_app - Aplikacja do rozpoznawania pojazdów.
"""

__version__ = "1.0.0"
__all__ = ['CarDetector', 'BrandClassifier', 'ANPRModule']


def __getattr__(name):
    """Leniwy import ciężkich modułów — ładowane dopiero przy pierwszym użyciu."""
    if name == 'CarDetector':
        from .detection import CarDetector
        return CarDetector
    if name == 'BrandClassifier':
        from .classification import BrandClassifier
        return BrandClassifier
    if name == 'ANPRModule':
        from .anpr import ANPRModule
        return ANPRModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
