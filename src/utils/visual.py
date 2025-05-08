import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def show_validation(img, detections, image_name, save_dir, title="Validação Visual"):
    if detections.empty:
        print("Nenhuma detecção para visualização.")
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    num_dets = min(len(detections), 5)
    fig, axs = plt.subplots(1, num_dets, figsize=(15, 5))
    if num_dets == 1:
        axs = [axs]

    for i, (index, row) in enumerate(detections.iterrows()):
        if i >= len(axs):
            break
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        crop = img[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        axs[i].imshow(crop_rgb)
        axs[i].axis('off')
        axs[i].set_title(f"{row['name']} ({row['confidence']:.2f})")

    plt.suptitle(title)
    plt.tight_layout()

    save_path = save_dir / f"val_{Path(image_name).stem}.png"
    plt.savefig(save_path)
    plt.close(fig)
