import torch
import cv2
import os
import time
import logging
from pathlib import Path
import pandas as pd

# Configurar logging
logging.basicConfig(
    filename='../../results/yolo/detection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Diretórios
DATA_DIR = Path('../../data')
RESULTS_DIR = Path('../../results/yolo')
CROPS_DIR = RESULTS_DIR / 'crops'  # Diretório para salvar os crops
CROPS_DIR.mkdir(parents=True, exist_ok=True)  # Criar o diretório se não existir
IMAGES = list(DATA_DIR.glob('*.jpeg'))

# Carregar modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.2  # confiança mínima

# Lista para métricas
metrics = []

def detect_image(image_path):
    start_time = time.time()
    img = cv2.imread(str(image_path))

    # ⬇️ Adicione esta verificação aqui
    if img is None:
        logging.error(f"{image_path.name} - Falha ao carregar imagem (img é None)")
        metrics.append({
            'imagem': image_path.name,
            'status': 'erro_leitura',
            'objetos_detectados': 0,
            'tempo_processamento_s': 0
        })
        return

    try:
        results = model(img)
        detections = results.pandas().xyxy[0]
        num_detections = len(detections)

        # Verifica sucesso
        status = 'sucesso' if num_detections > 0 else 'falha'
        logging.info(f"{image_path.name} - Status: {status} - Objetos: {num_detections}")

        # Salva imagem com boxes
        results.save(save_dir=str(RESULTS_DIR))

        # Gerar crops e salvar
        for i, row in detections.iterrows():
            # Coordenadas dos bounding boxes
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            
            # Realizar o crop da imagem
            crop = img[y1:y2, x1:x2]

            # Salvar o crop com um nome único
            crop_filename = CROPS_DIR / f"{image_path.stem}_crop_{i}.jpeg"
            cv2.imwrite(str(crop_filename), crop)
            logging.info(f"{image_path.name} - Crop {i} salvo como {crop_filename}")

            # Métricas por objeto
            category = row['name']
            width = x2 - x1
            height = y2 - y1
            area = width * height
            logging.info(f"{image_path.name} - Categoria: {category} - Tamanho: {area:.1f} px²")

        # Tempo de processamento
        elapsed = time.time() - start_time
        logging.info(f"{image_path.name} - Tempo: {elapsed:.2f}s")

        # Métrica geral da imagem
        metrics.append({
            'imagem': image_path.name,
            'status': status,
            'objetos_detectados': num_detections,
            'tempo_processamento_s': round(elapsed, 2)
        })

    except Exception as e:
        logging.error(f"{image_path.name} - Erro: {str(e)}")
        metrics.append({
            'imagem': image_path.name,
            'status': 'erro',
            'objetos_detectados': 0,
            'tempo_processamento_s': 0
        })

def main():
    logging.info("Início da detecção com YOLOv5")
    
    for image_path in IMAGES:
        print(f"Detectando: {image_path.name}")
        detect_image(image_path)
    
    # Salvar CSV com métricas gerais
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(RESULTS_DIR / 'metricas_yolo.csv', index=False)
    
    logging.info("Processamento finalizado.")

if __name__ == "__main__":
    main()
