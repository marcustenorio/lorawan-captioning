import torch
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import Delaunay
import pandas as pd
import string

# Caminhos
DATA_DIR = Path("data")
OUT_DIR = Path("results/yolo/graph_topology_planar_matrix")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Carrega o modelo YOLO
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.2

# Lista de imagens
image_paths = sorted(list(DATA_DIR.glob("*.jpg")) + list(DATA_DIR.glob("*.jpeg")) + list(DATA_DIR.glob("*.png")))

identificadores_base = list(string.ascii_uppercase)

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
    rotulos = []

    for idx, (_, row) in enumerate(detections.iterrows()):
        x1, y1, x2, y2 = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        letra = identificadores_base[idx % len(identificadores_base)]
        classe = row["name"].capitalize()
        rotulo = f"{letra} - {classe}"

        centers.append([cx, cy])
        rotulos.append(rotulo)

    points = np.array(centers)
    n = len(points)
    adjacency = np.zeros((n, n), dtype=int)

    if len(points) >= 3:
        tri = Delaunay(points)

        # Preenchendo matriz de adjacência com base nos triângulos
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    adjacency[a, b] = 1
                    adjacency[b, a] = 1  # matriz simétrica

    # Criação da matriz binária com labels amigáveis
    df_adj = pd.DataFrame(adjacency, columns=rotulos, index=rotulos)
    out_csv = OUT_DIR / f"{image_path.stem}_matriz_binaria.csv"
    df_adj.to_csv(out_csv)
    print(f"Matriz binária salva: {out_csv}")

    # (Opcional) Exibir no terminal
    print(df_adj)
