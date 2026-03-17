from ultralytics import YOLO
from pathlib import Path

model_path = Path(__file__).parent / 'car_vision_app' / 'anpr_best.pt'
print(f'Loading: {model_path}')
m = YOLO(str(model_path))
print('Task:', m.task)
print('Number of classes:', len(m.names))
print('Classes:', m.names)
