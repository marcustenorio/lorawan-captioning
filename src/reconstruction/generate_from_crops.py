import pandas as pd
from pathlib import Path
from diffusers import StableDiffusionPipeline
import torch
from tqdm import tqdm
import os
from huggingface_hub import login

# Autenticar com token (necessário no GitHub Actions)
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    login(token=hf_token)

# Diretórios
CSV_PATH = Path("results/blip/captions/captions_crops_detalhado.csv")
OUT_DIR = Path("results/reconstructions/from_crops")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Modelo de geração
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32  # use float16 se estiver em GPU local
).to("cuda" if torch.cuda.is_available() else "cpu")

# Carrega captions
df = pd.read_csv(CSV_PATH)

print(f"Gerando imagens a partir de {len(df)} captions de crops...")

for i, row in tqdm(df.iterrows(), total=len(df)):
    prompt = row["caption"]
    crop_name = Path(row["crop"]).stem

    try:
        img = pipe(prompt).images[0]
        img.save(OUT_DIR / f"reconstructed_detalhado_crop_{crop_name}.png")
    except Exception as e:
        print(f"[Erro] {crop_name}: {e}")

print(f"Imagens geradas em: {OUT_DIR}")

