import logging
from pathlib import Path
import pytz
from datetime import datetime

# Função para formatar o timestamp no fuso horário GMT-3 (Brasília)
class GMT3Formatter(logging.Formatter):
    def convert_to_gmt3(self, record):
        # Define o fuso horário como Brasilia (GMT-3)
        brasilia_tz = pytz.timezone('America/Sao_Paulo')
        # Converte o horário para o fuso horário correto
        record.asctime = datetime.fromtimestamp(record.created, brasilia_tz).strftime('%Y-%m-%d %H:%M:%S')
        return record

    def format(self, record):
        # Adiciona a data/hora no formato desejado
        record = self.convert_to_gmt3(record)
        return super().format(record)

def setup_logging():
    log_path = Path('results/yolo/detection.log').resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Cria o log com a data e hora no formato GMT-3
    formatter = GMT3Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'  # Data no formato YYYY-MM-DD HH:MM:SS
    )

    handler = logging.FileHandler(log_path)
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def log_object_metrics(image_name, detections):
    for i, row in detections.iterrows():
        category = row['name']
        width = row['xmax'] - row['xmin']
        height = row['ymax'] - row['ymin']
        area = width * height
        logging.info(f"{image_name} - Categoria: {category} - Tamanho: {area:.1f} px²")
