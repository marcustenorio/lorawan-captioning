import cv2
import os
import time
import logging
import pandas as pd
from pathlib import Path

# Detectron2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Configurar logging
RESULTS_DIR = Path("results/detectron2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=RESULTS_DIR / 'detection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Lista de imagens
DATA_DIR = Path("../../data")
IMAGES = list(DATA_DIR.glob("*.jpeg"))

# Configurar modelo Detectron2 (Faster R-CNN)
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

# Métricas
metrics = []

def detect_image(image_path):
    start_time = time.time()
    img = cv2.imread(str(image_path))

    if img is None:
        logging.error(f"{image_path.name} - Erro ao carregar imagem.")
        metrics.append({
            'imagem': image_path.name,
            'status': 'erro_leitura',
            'objetos_detectados': 0,
            'tempo_processamento_s': 0
        })
        return

    try:
        outputs = predictor(img)
        instances = outputs["instances"]
        pred_classes = instances.pred_classes
        pred_boxes = instances.pred_boxes.tensor

        num_detections = len(pred_classes)
        status = 'sucesso' if num_detections > 0 else 'falha'
        logging.info(f"{image_path.name} - Status: {status} - Objetos: {num_detections}")

        # Logs por objeto
        for cls_id, box in zip(pred_classes, pred_boxes):
            class_name = metadata.get("thing_classes", [])[cls_id]
            width = box[2] - box[0]
            height = box[3] - box[1]
            area = width * height
            logging.info(f"{image_path.name} - Categoria: {class_name} - Tamanho: {area:.1f} px²")

        # Visualização
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        vis_output = visualizer.draw_instance_predictions(instances.to("cpu"))
        output_path = RESULTS_DIR / f"boxed_{image_path.name}"
        cv2.imwrite(str(output_path), vis_output.get_image()[:, :, ::-1])

        # Tempo
        elapsed = time.time() - start_time
        logging.info(f"{image_path.name} - Tempo: {elapsed:.2f}s")

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
    logging.info("Início da detecção com Detectron2")
    for image_path in IMAGES:
        print(f"Detectando: {image_path.name}")
        detect_image(image_path)
    
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(RESULTS_DIR / 'metricas_detectron2.csv', index=False)
    logging.info("Processamento finalizado.")

if __name__ == "__main__":
    main()

