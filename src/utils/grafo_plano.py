import torch
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import Delaunay
import pandas as pd
import string
from matplotlib.cm import get_cmap

# Caminhos
DATA_DIR = Path("data")
OUT_DIR_IMG = Path("results/yolo/graph_topology_planar")
OUT_DIR_CSV = Path("results/yolo/graph_topology_planar_descriptors")
OUT_DIR_IMG.mkdir(parents=True, exist_ok=True)
OUT_DIR_CSV.mkdir(parents=True, exist_ok=True)

# Carrega modelo YOLO
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.2

# Lista de imagens
image_paths = sorted(list(DATA_DIR.glob("*.jpg")) + list(DATA_DIR.glob("*.jpeg")) + list(DATA_DIR.glob("*.png")))

# Mapeamento de cores
cmap = get_cmap("tab20")

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
    cores_por_classe = {}
    identificadores = list(string.ascii_uppercase)

    for idx, (_, row) in enumerate(detections.iterrows()):
        x1, y1, x2, y2 = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        area = (x2 - x1) * (y2 - y1)

        classe = row["name"].capitalize()
        letra = identificadores[idx % len(identificadores)]

        # Cor única por classe
        if classe not in cores_por_classe:
            cores_por_classe[classe] = cmap(len(cores_por_classe))

        centers.append([cx, cy])
        labels.append((letra, classe))

        descritores.append({
            "identificador": letra,
            "imagem": image_path.name,
            "classe": classe,
            "confiança": round(row["confidence"], 4),
            "centro_x": round(cx, 2),
            "centro_y": round(cy, 2),
            "area_px": round(area, 2)
        })

    points = np.array(centers)

    # Plot do grafo planar
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(f"Grafo Planar das Detecções – {image_path.name}", fontsize=14)
    ax.set_xlabel("Coordenada X (pixels)", fontsize=12)
    ax.set_ylabel("Coordenada Y (pixels)", fontsize=12)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Plotando pontos com rótulos dispersos
    for idx, ((x, y), (letra, classe)) in enumerate(zip(points, labels)):
        cor = cores_por_classe[classe]
        ax.plot(x, y, 'o', markersize=10, color=cor)

        # Deslocamento em círculo alternado para evitar sobreposição
        dx = 10 * np.cos(idx * np.pi / 6)
        dy = -10 * np.sin(idx * np.pi / 6)
        ax.text(x + dx, y + dy, f"{letra} - {classe}", fontsize=9, color=cor, fontweight='bold')

    if len(points) >= 3:
        tri = Delaunay(points)
        for simplex in tri.simplices:
            pts = points[simplex]
            ax.plot(pts[:, 0], pts[:, 1], 'gray', alpha=0.5)

    # Legenda
    legend_handles = []
    for classe, cor in cores_por_classe.items():
        handle = plt.Line2D([0], [0], marker='o', color='w', label=classe,
                            markerfacecolor=cor, markersize=10)
        legend_handles.append(handle)
    ax.legend(handles=legend_handles, title="Classes detectadas")

    plt.tight_layout()
    out_img = OUT_DIR_IMG / f"{image_path.stem}_grafo_planar.png"
    plt.savefig(out_img)
    plt.close()
    print(f"Grafo planar salvo: {out_img}")

    # Salva CSV
    df_descritores = pd.DataFrame(descritores)
    out_csv = OUT_DIR_CSV / f"{image_path.stem}_descritores.csv"
    df_descritores.to_csv(out_csv, index=False)
    print(f"Descritores salvos: {out_csv}")
