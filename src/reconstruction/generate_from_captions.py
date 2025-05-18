import pandas as pd
from pathlib import Path
from diffusers import StableDiffusionPipeline
import torch
from tqdm import tqdm

import os
from huggingface_hub import login

# Autentica com o token do GitHub Actions
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    login(token=hf_token)


# Configurações
#CSV_PATH = Path("results/blip/full_image/captions_full_image.csv")
CSV_PATH = Path("results/blip/full_image/full_image_detalhado.csv")
OUT_DIR = Path("results/reconstructions/from_caption")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Carrega modelo de difusão
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float32
).to("cuda" if torch.cuda.is_available() else "cpu")

# Lê os captions
df = pd.read_csv(CSV_PATH)

print(f"Gerando imagens para {len(df)} descrições...")

for i, row in tqdm(df.iterrows(), total=len(df)):
    prompt = row["caption"]
    image_name = Path(row["imagem"]).stem
    try:
        img = pipe(prompt).images[0]
        img.save(OUT_DIR / f"reconstructed_detalhado_{image_name}.png")
    except Exception as e:
        print(f"[Erro] ao gerar '{image_name}': {e}")

print(f"Imagens salvas em: {OUT_DIR}")
