import cv2
import logging
from pathlib import Path

def save_crops(img, detections, image_path, crops_dir: Path):
    for i, row in detections.iterrows():
        x1, y1 = int(row['xmin']), int(row['ymin'])
        x2, y2 = int(row['xmax']), int(row['ymax'])
        crop = img[y1:y2, x1:x2]
        crop_filename = crops_dir / f"{image_path.stem}_crop_{i}.jpeg"
        cv2.imwrite(str(crop_filename), crop)
        logging.info(f"{image_path.name} - Crop {i} salvo como {crop_filename}")
