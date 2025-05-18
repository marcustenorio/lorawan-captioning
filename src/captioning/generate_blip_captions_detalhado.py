from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import pandas as pd

#  Diretórios
CROPS_DIR = Path("results/yolo/crops")
OUT_DIR = Path("results/blip/captions")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "captions_crops_detalhado.csv"

#  BLIP model
print("🔧 Carregando BLIP large...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
model.eval()

#  Carrega todos os crops
crop_paths = sorted(list(CROPS_DIR.glob("*.jpg")) + list(CROPS_DIR.glob("*.JPG")) + list(CROPS_DIR.glob("*.png")) + list(CROPS_DIR.glob("*.jpeg")))

print(f"Gerando descrições detalhadas para {len(image_paths)} crops...")

#  Resultados
captions = []

for img_path in tqdm(image_paths):
    try:
        image = Image.open(img_path).convert("RGB")

        # Prompt que estimula mais riqueza semântica
        prompt = "Describe this object in detail, including its color, texture, shape, and any visible context."

        inputs = processor(image, prompt, return_tensors="pt")
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2
        )

        caption = processor.decode(output[0], skip_special_tokens=True)

        captions.append({
            "crop": img_path.name,
            "caption": caption
        })

    except Exception as e:
        print(f"[Erro] {img_path.name}: {e}")

#  Salva CSV
df = pd.DataFrame(captions)
df.to_csv(OUT_CSV, index=False)

print(f"Captions detalhadas salvas em: {OUT_CSV}")

