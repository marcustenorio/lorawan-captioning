import clip
import torch
from PIL import Image
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# 📁 Caminhos
ORIGINAL_DIR = Path("data")
RECONSTRUCTED_DIR = Path("results/reconstructions/from_caption")
CAPTION_CSV = Path("results/blip/full_image/captions_full_image.csv")
OUT_CSV = Path("results/reconstructions/clip_similarity.csv")

# 🧠 Carrega CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 📄 Lê captions
df = pd.read_csv(CAPTION_CSV)

resultados = []

print(f"Comparando imagens originais vs. reconstruídas com CLIP...")

for i, row in tqdm(df.iterrows(), total=len(df)):
    nome_imagem = row["imagem"]
    caption = row["caption"]

    try:
        original_path = ORIGINAL_DIR / nome_imagem
        reconstruida_path = RECONSTRUCTED_DIR / f"reconstructed_{Path(nome_imagem).stem}.png"

        if not original_path.exists() or not reconstruida_path.exists():
            continue

        img1 = preprocess(Image.open(original_path).convert("RGB")).unsqueeze(0).to(device)
        img2 = preprocess(Image.open(reconstruida_path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            feat1 = model.encode_image(img1)
            feat2 = model.encode_image(img2)

            score = torch.nn.functional.cosine_similarity(feat1, feat2).item()

        resultados.append({
            "imagem": nome_imagem,
            "caption": caption,
            "score_clip": round(score, 4)
        })

    except Exception as e:
        print(f"[Erro] {nome_imagem}: {e}")

# 💾 Salvar resultado
df_out = pd.DataFrame(resultados)
df_out.to_csv(OUT_CSV, index=False)

print(f"Similaridade CLIP salva em: {OUT_CSV}")
