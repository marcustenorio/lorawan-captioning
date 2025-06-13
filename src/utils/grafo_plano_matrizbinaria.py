import torch
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import Delaunay
import pandas as pd

DATA_DIR = Path("data")
OUT_DIR = Path("results/yolo/graph_topology_planar_matrix")
OUT_DIR.mkdir(parents=True, exist_ok=True)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.2

image_paths = sorted(list(DATA_DIR.glob("*.jpg")) + list(DATA_DIR.glob("*.jpeg")) + list(DATA_DIR.glob("*.png")))

for image_path in image_paths:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[⚠️] Erro ao ler: {image_path.name}")
        continue

    results = model(img)
    detections = results.pandas().xyxy[0]

    if detections.empty:
        print(f"[⚠️] Nenhuma detecção em: {image_path.name}")
        continue

    centers = []
    labels = []

    for _, row in detections.iterrows():
        x1, y1, x2, y2 = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centers.append([cx, cy])
        labels.append(row["name"])

    points = np.array(centers)
    n = len(points)
    adjacency = np.zeros((n, n), dtype=int)

    if len(points) >= 3:
        tri = Delaunay(points)

        # Para cada triângulo, conecta os vértices
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    adjacency[a, b] = 1
                    adjacency[b, a] = 1  # matriz simétrica

    # Salvar a matriz como CSV binário
    df_adj = pd.DataFrame(adjacency, columns=[f"obj_{i}" for i in range(n)], index=[f"obj_{i}" for i in range(n)])
    out_csv = OUT_DIR / f"{image_path.stem}_matriz_binaria.csv"
    df_adj.to_csv(out_csv)
    print(f"✅ Matriz binária salva: {out_csv}")

    # (Opcional) Mostrar matriz no console
    print(df_adj)
