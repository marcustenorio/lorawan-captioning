from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import pandas as pd

# Diretórios
IMAGES_DIR = Path("data")  # ou "data" se for full images
OUT_CSV = Path("results/blip/full_image/full_image_detalhado.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# Carrega modelo BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
model.eval()

# 🔍 Lista imagens
IMAGES = list(IMAGES_DIR.glob("*.png")) + list(IMAGES_DIR.glob("*.JPG")) + list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.jpeg"))

# Prompt detalhado
PROMPT = "Describe this image with as much detail as possible, including objects, colors, positions, and actions."

# Resultados
captions = []

print(f"Gerando descrições detalhadas para {len(IMAGES)} imagens...")

for img_path in tqdm(IMAGES):
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(image, text=PROMPT, return_tensors="pt")

        out = model.generate(**inputs, max_new_tokens=100, num_beams=5, early_stopping=True)
        caption = processor.decode(out[0], skip_special_tokens=True)

        captions.append({
            "imagem": img_path.name,
            "caption": caption
        })

    except Exception as e:
        print(f"[Erro] {img_path.name}: {e}")

# Salva CSV
df = pd.DataFrame(captions)
df.to_csv(OUT_CSV, index=False)
print(f"Captions detalhadas salvas em: {OUT_CSV}")
