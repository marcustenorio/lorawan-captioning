import torch
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

# Caminhos
DATA_DIR = Path("data")
OUT_DIR = Path("results/yolo/graph_topology")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Modelo YOLO
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.2

# Imagens
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

    # Extrai centros dos bounding boxes
    centers = []
    labels = []

    for _, row in detections.iterrows():
        x1, y1, x2, y2 = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centers.append((cx, cy))
        labels.append(row["name"])

    # Plotagem como grafo/plano
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(f"Topologia dos objetos – {image_path.name}")
    ax.set_xlabel("Posição X (px)")
    ax.set_ylabel("Posição Y (px)")
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)  # Inverter eixo Y para bater com coordenadas de imagem

    for (x, y), label in zip(centers, labels):
        ax.plot(x, y, 'o', markersize=10, label=label, alpha=0.7)
        ax.text(x + 5, y - 5, label, fontsize=9, color='black')

    # Opcional: conectividade (arestas entre vizinhos)
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            xi, yi = centers[i]
            xj, yj = centers[j]
            dist = ((xi - xj)**2 + (yi - yj)**2)**0.5
            if dist < 300:  # limiar de vizinhança ajustável
                ax.plot([xi, xj], [yi, yj], 'k--', linewidth=0.8, alpha=0.3)

    plt.tight_layout()
    save_path = OUT_DIR / f"{image_path.stem}_grafo_topologia.png"
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Grafo/topologia gerado: {save_path}")
