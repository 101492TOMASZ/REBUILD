"""
Moduł klasyfikacji marki pojazdu przy użyciu MobileNetV2.
Odpowiada za rozpoznawanie marki samochodu na wyciętym obrazie.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class BrandClassifier:
    """Klasa do klasyfikacji marki pojazdu przy użyciu MobileNetV2."""
    
    def __init__(self, model_path: str, classes_path: str, device: str = None):
        """
        Inicjalizacja klasyfikatora marki.
        
        Args:
            model_path: Ścieżka do pliku z wagami modelu (.pth)
            classes_path: Ścieżka do pliku JSON z mapowaniem klas
            device: Urządzenie do obliczeń ('cuda' lub 'cpu')
        """
        # Automatyczny wybór urządzenia
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Wczytaj mapowanie klas
        self.classes = self._load_classes(classes_path)
        self.num_classes = len(self.classes)
        
        # Utwórz model
        self.model = self._create_model()
        
        # Wczytaj wagi
        self._load_weights(model_path)
        
        # Transformacje dla obrazu wejściowego
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_classes(self, classes_path: str) -> dict:
        """
        Wczytuje mapowanie klas z pliku JSON.
        
        Args:
            classes_path: Ścieżka do pliku JSON
            
        Returns:
            Słownik {nazwa_klasy: indeks}
        """
        with open(classes_path, 'r') as f:
            classes = json.load(f)
        
        # Utwórz odwrotne mapowanie (indeks -> nazwa)
        self.idx_to_class = {v: k for k, v in classes.items()}
        return classes
    
    def _create_model(self) -> nn.Module:
        """
        Tworzy architekturę MobileNetV2 z odpowiednią liczbą klas wyjściowych.
        
        Returns:
            Model MobileNetV2
        """
        # Utwórz model MobileNetV2 bez pretrenowanych wag
        model = mobilenet_v2(weights=None)
        
        # Zmodyfikuj klasyfikator dla odpowiedniej liczby klas
        # Oryginalny klasyfikator MobileNetV2:
        # classifier = Sequential(
        #     Dropout(p=0.2),
        #     Linear(in_features=1280, out_features=1000)
        # )
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, self.num_classes)
        )
        
        return model.to(self.device)
    
    def _load_weights(self, model_path: str):
        """
        Wczytuje wagi modelu z pliku.
        
        Args:
            model_path: Ścieżka do pliku z wagami
        """
        # weights_only=False jest wymagane dla modeli zapisanych w starszym formacie
        # Używaj tylko z zaufanych źródeł!
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Obsługa różnych formatów zapisu wag
        if isinstance(checkpoint, dict):
            if 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Próba załadowania wag
        try:
            self.model.load_state_dict(state_dict)
            print(f"Model załadowany pomyślnie na urządzenie: {self.device}")
        except RuntimeError as e:
            # Jeśli klucze nie pasują, spróbuj dopasować
            print(f"Ostrzeżenie: Dopasowywanie kluczy wag... ({e})")
            model_dict = self.model.state_dict()
            # Filtruj tylko pasujące klucze
            filtered_dict = {k: v for k, v in state_dict.items() 
                           if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(filtered_dict)
            self.model.load_state_dict(model_dict)
            print(f"Model załadowany (częściowo) na urządzenie: {self.device}")
        
        self.model.eval()
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Przetwarza obraz wejściowy do formatu wymaganego przez model.
        
        Args:
            image: Obraz BGR (OpenCV)
            
        Returns:
            Tensor gotowy do podania na wejście modelu
        """
        # Konwersja BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Zastosuj transformacje
        tensor = self.transform(image_rgb)
        
        # Dodaj wymiar batch
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Klasyfikuje markę pojazdu na obrazie.
        
        Args:
            image: Obraz BGR z wyciętym pojazdem
            
        Returns:
            Tuple (nazwa_marki, pewność_procentowa)
        """
        # Przetwórz obraz
        input_tensor = self.preprocess(image)
        
        # Predykcja
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Pobierz nazwę klasy
        predicted_class = self.idx_to_class[predicted_idx.item()]
        confidence_percent = confidence.item() * 100
        
        return predicted_class, confidence_percent
    
    def predict_top_k(self, image: np.ndarray, k: int = 3) -> list:
        """
        Zwraca top-k predykcji dla obrazu.
        
        Args:
            image: Obraz BGR z wyciętym pojazdem
            k: Liczba najlepszych predykcji do zwrócenia
            
        Returns:
            Lista krotek (nazwa_marki, pewność_procentowa)
        """
        # Przetwórz obraz
        input_tensor = self.preprocess(image)
        
        # Predykcja
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, min(k, self.num_classes))
        
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            class_name = self.idx_to_class[idx.item()]
            confidence = prob.item() * 100
            results.append((class_name, confidence))
        
        return results
    
    def generate_gradcam(self, image: np.ndarray, target_class: int = None) -> np.ndarray:
        """
        Generuje mapę ciepła Grad-CAM pokazującą które obszary obrazu
        wpłynęły na decyzję klasyfikatora.
        
        Args:
            image: Obraz BGR z wyciętym pojazdem
            target_class: Indeks klasy docelowej (None = predykcja modelu)
            
        Returns:
            Mapa ciepła nałożona na oryginalny obraz (BGR)
        """
        # Przetwórz obraz
        input_tensor = self.preprocess(image)
        input_tensor.requires_grad_(True)
        
        # Rejestruj hooki dla ostatniej warstwy konwolucyjnej
        # W MobileNetV2 to features[-1] (ostatni blok)
        activations = []
        gradients = []
        
        def forward_hook(module, input, output):
            activations.append(output)
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        # Zarejestruj hooki na ostatniej warstwie konwolucyjnej
        target_layer = self.model.features[-1]
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Jeśli nie podano klasy docelowej, użyj predykcji
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass dla wybranej klasy
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        # Usuń hooki
        forward_handle.remove()
        backward_handle.remove()
        
        # Oblicz wagi (global average pooling gradientów)
        grad = gradients[0]  # [1, C, H, W]
        weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # Ważona suma aktywacji
        act = activations[0]  # [1, C, H, W]
        cam = torch.sum(weights * act, dim=1, keepdim=True)  # [1, 1, H, W]
        
        # ReLU (tylko pozytywne wpływy)
        cam = F.relu(cam)
        
        # Normalizacja
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Konwersja do numpy i resize do rozmiaru obrazu
        cam = cam.squeeze().cpu().detach().numpy()
        cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
        
        # Konwersja do mapy ciepła
        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Nałóż na oryginalny obraz
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        
        return overlay, heatmap, cam
