import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Inicializa BLIP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Diretório com crops gerados pelo YOLO
CROPS_DIR = Path("results/yolo/crops")
OUTPUT_DIR = Path("results/blip/captions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Lista todos os crops válidos
crop_paths = sorted(list(CROPS_DIR.glob("*.jpg")) + list(CROPS_DIR.glob("*.JPG")) + list(CROPS_DIR.glob("*.png")) + list(CROPS_DIR.glob("*.jpeg")))

# Processa cada crop
captions_data = []

print(f"🔍 Gerando captions para {len(crop_paths)} crops...")

for crop_path in tqdm(crop_paths):
    try:
        image = Image.open(crop_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        captions_data.append({
            "crop": crop_path.name,
            "imagem_original": "_".join(crop_path.stem.split("_")[:-1]) + ".jpg",
            "caption": caption
        })

    except Exception as e:
        print(f"[ERRO] {crop_path.name}: {e}")

# 💾 Salva como CSV
df = pd.DataFrame(captions_data)
df.to_csv(OUTPUT_DIR / "captions_crops.csv", index=False)

# (Opcional) Salvar JSON
df.to_json(OUTPUT_DIR / "captions_crops.json", orient="records", indent=2)

print("Captions salvos em:", OUTPUT_DIR)
