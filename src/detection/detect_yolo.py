import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import cv2
import time
import pandas as pd
from src.utils.crops import save_crops
from src.utils.logger import log_object_metrics, setup_logging
from src.utils.metrics import log_image_metrics, metrics

# Inicializar logging
setup_logging()

# Diretórios
DATA_DIR = Path('data')
RESULTS_DIR = Path('results/yolo')
CROPS_DIR = RESULTS_DIR / 'crops'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES = list(DATA_DIR.glob('*.jpeg'))

# Carrega o modelo
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.2


def detect_image(image_path):
    start_time = time.time()
    img = cv2.imread(str(image_path))

    if img is None:
        print(f"Erro ao ler imagem: {image_path}")
        log_image_metrics(image_path.name, 'erro_leitura', 0, 0)
        return

    try:
        results = model(img)
        detections = results.pandas().xyxy[0]
        num_detections = len(detections)
        status = 'sucesso' if num_detections > 0 else 'falha'

        results.save(save_dir=str(RESULTS_DIR))

        if num_detections > 0:
            save_crops(img, detections, image_path, CROPS_DIR)
            log_object_metrics(image_path.name, detections)

        elapsed = time.time() - start_time
        log_image_metrics(image_path.name, status, num_detections, elapsed)

    except Exception as e:
        print(f"Erro na imagem {image_path.name}: {e}")
        log_image_metrics(image_path.name, 'erro', 0, 0)


def main():
    print("Início da detecção com YOLOv5")

    for image_path in IMAGES:
        print(f"Detectando: {image_path.name}")
        detect_image(image_path)

    df = pd.DataFrame(metrics)
    df.to_csv(RESULTS_DIR / 'metricas_yolo.csv', index=False)

    print("Processamento finalizado.")


if __name__ == "__main__":
    main()
