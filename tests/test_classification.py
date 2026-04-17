"""
Testy jednostkowe modułu klasyfikacji marki pojazdu (BrandClassifier).

Testy jednostkowe mockują model PyTorch.
Testy integracyjne (oznaczone @pytest.mark.integration) wymagają prawdziwych
plików model.pth i label_map.json.
"""

import sys
import pytest
import numpy as np
import torch
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import requires_models, MODEL_PATH, CLASSES_PATH


# ============================================================
# Fixture — BrandClassifier bez prawdziwego modelu
# ============================================================

@pytest.fixture
def classifier_bare(tmp_path):
    """
    BrandClassifier z zamockowanym modelem i prawdziwym label_map.json.
    Używa temp katalogu na sztuczny plik klas.
    """
    from car_vision_app.classification import BrandClassifier

    labels = {"AUDI": 0, "BMW": 1, "MERCEDES": 2, "PORSCHE": 3, "VOLKSWAGEN": 4}
    classes_file = tmp_path / "label_map.json"
    classes_file.write_text(json.dumps(labels))

    fake_model_file = tmp_path / "model.pth"
    fake_weights = {f"features.{i}.weight": torch.zeros(1) for i in range(3)}
    torch.save(fake_weights, str(fake_model_file))

    with patch.object(BrandClassifier, '_create_model') as mock_create, \
         patch.object(BrandClassifier, '_load_weights'):
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_create.return_value = mock_model

        clf = BrandClassifier(
            model_path=str(fake_model_file),
            classes_path=str(classes_file),
        )
        clf.model = mock_model

    return clf, labels


# ============================================================
# Testy ładowania klas
# ============================================================

class TestLoadClasses:
    def test_classes_loaded(self, classifier_bare):
        clf, labels = classifier_bare
        assert clf.classes == labels

    def test_idx_to_class_inverted(self, classifier_bare):
        clf, labels = classifier_bare
        for name, idx in labels.items():
            assert clf.idx_to_class[idx] == name

    def test_num_classes_correct(self, classifier_bare):
        clf, labels = classifier_bare
        assert clf.num_classes == len(labels)


# ============================================================
# Testy preprocess
# ============================================================

class TestPreprocess:
    def test_output_is_tensor(self, classifier_bare):
        clf, _ = classifier_bare
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        tensor = clf.preprocess(img)
        assert isinstance(tensor, torch.Tensor)

    def test_output_shape_batch1(self, classifier_bare):
        clf, _ = classifier_bare
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        tensor = clf.preprocess(img)
        # Batch=1, C=3, H=224, W=224
        assert tensor.shape == (1, 3, 224, 224)

    def test_output_dtype_float(self, classifier_bare):
        clf, _ = classifier_bare
        img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        tensor = clf.preprocess(img)
        assert tensor.dtype == torch.float32

    def test_small_image_resized_correctly(self, classifier_bare):
        clf, _ = classifier_bare
        img = np.ones((32, 32, 3), dtype=np.uint8) * 128
        tensor = clf.preprocess(img)
        assert tensor.shape[-2:] == (224, 224)

    def test_large_image_resized_correctly(self, classifier_bare):
        clf, _ = classifier_bare
        img = np.ones((1200, 1920, 3), dtype=np.uint8) * 200
        tensor = clf.preprocess(img)
        assert tensor.shape[-2:] == (224, 224)


# ============================================================
# Testy predict
# ============================================================

class TestPredict:
    def test_returns_tuple(self, classifier_bare):
        clf, labels = classifier_bare
        probs = torch.softmax(torch.tensor([[0.1, 5.0, 0.2, 0.3, 0.1]]), dim=1)
        clf.model.return_value = torch.tensor([[0.1, 5.0, 0.2, 0.3, 0.1]])

        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = clf.predict(img)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_predicted_class_is_string(self, classifier_bare):
        clf, labels = classifier_bare
        clf.model.return_value = torch.tensor([[0.1, 8.0, 0.2, 0.3, 0.1]])

        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        brand, _ = clf.predict(img)
        assert isinstance(brand, str)
        assert brand in labels

    def test_confidence_is_percentage(self, classifier_bare):
        clf, _ = classifier_bare
        clf.model.return_value = torch.tensor([[0.0, 0.0, 10.0, 0.0, 0.0]])

        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        _, confidence = clf.predict(img)
        assert 0.0 <= confidence <= 100.0

    def test_highest_logit_wins(self, classifier_bare):
        clf, _ = classifier_bare
        # BMW ma najwyższy logit (idx=1)
        clf.model.return_value = torch.tensor([[1.0, 15.0, 2.0, 0.5, 0.5]])

        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        brand, _ = clf.predict(img)
        assert brand == "BMW"

    def test_all_brand_outputs_valid(self, classifier_bare):
        clf, labels = classifier_bare

        for idx in range(len(labels)):
            logits = torch.zeros(1, len(labels))
            logits[0, idx] = 10.0
            clf.model.return_value = logits

            img = np.ones((224, 224, 3), dtype=np.uint8) * 100
            brand, confidence = clf.predict(img)
            assert brand in labels
            assert confidence > 50.0


# ============================================================
# Testy tworzenia modelu
# ============================================================

class TestCreateModel:
    def test_model_has_correct_output_classes(self, tmp_path):
        from car_vision_app.classification import BrandClassifier
        import torch.nn as nn

        labels = {"AUDI": 0, "BMW": 1, "MERCEDES": 2}
        classes_file = tmp_path / "label_map.json"
        classes_file.write_text(json.dumps(labels))

        with patch.object(BrandClassifier, '_load_weights'):
            clf = BrandClassifier.__new__(BrandClassifier)
            clf.device = torch.device('cpu')
            clf.classes = labels
            clf.idx_to_class = {v: k for k, v in labels.items()}
            clf.num_classes = len(labels)
            model = clf._create_model()

        # Sprawdź warstwę wyjściową
        classifier_layer = model.classifier[1]
        assert isinstance(classifier_layer, nn.Linear)
        assert classifier_layer.out_features == 3


# ============================================================
# Testy integracyjne (wymagają prawdziwych plików)
# ============================================================

@requires_models
@pytest.mark.integration
class TestBrandClassifierIntegration:
    @pytest.fixture
    def classifier(self):
        from car_vision_app.classification import BrandClassifier
        return BrandClassifier(
            model_path=str(MODEL_PATH),
            classes_path=str(CLASSES_PATH),
        )

    def test_model_loads_successfully(self, classifier):
        assert classifier.model is not None

    def test_predict_returns_known_brand(self, classifier):
        img = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
        brand, confidence = classifier.predict(img)
        known_brands = {"AUDI", "BMW", "MERCEDES", "PORSCHE", "VOLKSWAGEN"}
        assert brand in known_brands

    def test_predict_confidence_in_range(self, classifier):
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        _, confidence = classifier.predict(img)
        assert 0.0 <= confidence <= 100.0

    def test_gradcam_generates_heatmap(self, classifier):
        img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        overlay, heatmap, cam = classifier.generate_gradcam(img)
        assert overlay is not None
        assert heatmap is not None
        assert overlay.shape == img.shape
