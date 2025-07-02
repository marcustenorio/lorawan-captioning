import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from pathlib import Path
import pandas as pd
from tqdm import tqdm
#Gerando o prompt 

# Inicializa BLIP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Diretório com imagens originais
IMAGE_DIR = Path("data")
OUTPUT_DIR = Path("results/blip/full_image")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Lista imagens válidas
image_paths = sorted(list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.JPG")) + list(IMAGE_DIR.glob("*.jpeg")) + list(IMAGE_DIR.glob("*.png")))

captions_data = []

print(f"Gerando captions para {len(image_paths)} imagens completas...")

for image_path in tqdm(image_paths):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        captions_data.append({
            "imagem": image_path.name,
            "caption": caption
        })

    except Exception as e:
        print(f"[ERRO] {image_path.name}: {e}")

# 💾 Salva CSV e JSON
df = pd.DataFrame(captions_data)
df.to_csv(OUTPUT_DIR / "captions_full_image.csv", index=False)
df.to_json(OUTPUT_DIR / "captions_full_image.json", orient="records", indent=2)

print(f"Captions salvas em: {OUTPUT_DIR}")
