import pandas as pd
from pathlib import Path
from diffusers import StableDiffusionPipeline
import torch
from tqdm import tqdm

# 🔧 Configurações
CSV_PATH = Path("results/blip/full_image/captions_full_image.csv")
OUT_DIR = Path("results/reconstructions/from_caption")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Carrega modelo de difusão
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float16,
    revision="fp16"
).to("cuda" if torch.cuda.is_available() else "cpu")

# 📄 Lê os captions
df = pd.read_csv(CSV_PATH)

print(f"Gerando imagens para {len(df)} descrições...")

for i, row in tqdm(df.iterrows(), total=len(df)):
    prompt = row["caption"]
    image_name = Path(row["imagem"]).stem
    try:
        img = pipe(prompt).images[0]
        img.save(OUT_DIR / f"reconstructed_{image_name}.png")
    except Exception as e:
        print(f"[Erro] ao gerar '{image_name}': {e}")

print(f"Imagens salvas em: {OUT_DIR}")
