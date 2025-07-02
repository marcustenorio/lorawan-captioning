import torch
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import Delaunay
import pandas as pd

# Caminhos
DATA_DIR = Path("data")
OUT_DIR_IMG = Path("results/yolo/graph_topology_planar")
OUT_DIR_CSV = Path("results/yolo/graph_topology_planar_descriptors")
OUT_DIR_IMG.mkdir(parents=True, exist_ok=True)
OUT_DIR_CSV.mkdir(parents=True, exist_ok=True)

# Modelo YOLO
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.2

# Imagens
image_paths = sorted(list(DATA_DIR.glob("*.jpg")) + list(DATA_DIR.glob("*.jpeg")) + list(DATA_DIR.glob("*.png")))

for image_path in image_paths:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Erro ao ler: {image_path.name}")
        continue

    results = model(img)
    detections = results.pandas().xyxy[0]

    if detections.empty:
        print(f"Nenhuma detecção em: {image_path.name}")
        continue

    centers = []
    labels = []
    descritores = []

    for _, row in detections.iterrows():
        x1, y1, x2, y2 = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        area = (x2 - x1) * (y2 - y1)

        centers.append([cx, cy])
        labels.append(row["name"])

        descritores.append({
            "imagem": image_path.name,
            "classe": row["name"],
            "confiança": round(row["confidence"], 4),
            "centro_x": round(cx, 2),
            "centro_y": round(cy, 2),
            "area_px": round(area, 2)
        })

    points = np.array(centers)

    # Plot plano topológico
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(f"Plano topológico (grafo planar) – {image_path.name}")
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.grid(True, linestyle="--", alpha=0.3)

    ax.plot(points[:, 0], points[:, 1], 'o', markersize=10)

    for (x, y), label in zip(points, labels):
        ax.text(x + 5, y - 5, label, fontsize=9)

    if len(points) >= 3:
        tri = Delaunay(points)
        for simplex in tri.simplices:
            pts = points[simplex]
            ax.plot(pts[:, 0], pts[:, 1], 'k-', alpha=0.5)

    plt.tight_layout()
    out_img = OUT_DIR_IMG / f"{image_path.stem}_grafo_planar.png"
    plt.savefig(out_img)
    plt.close()
    print(f"Grafo planar salvo: {out_img}")

    # Salva descritores
    df_descritores = pd.DataFrame(descritores)
    out_csv = OUT_DIR_CSV / f"{image_path.stem}_descritores.csv"
    df_descritores.to_csv(out_csv, index=False)
    print(f"Descritores salvos: {out_csv}")
