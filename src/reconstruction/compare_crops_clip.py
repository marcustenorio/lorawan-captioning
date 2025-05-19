import clip
import torch
from PIL import Image
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# 📁 Caminhos
ORIGINAL_CROPS_DIR = Path("results/yolo/crops")
RECONSTRUCTED_CROPS_DIR = Path("results/reconstructions/from_crops")
CAPTION_CSV = Path("results/blip/captions/captions_crops_detalhado.csv")
OUT_CSV = Path("results/reconstructions/clip_similarity_crops.csv")

# 🧠 Carrega CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 📄 Lê captions detalhados
df = pd.read_csv(CAPTION_CSV)

resultados = []

print(f"Comparando crops originais vs. crops reconstruídos com CLIP...")

for i, row in tqdm(df.iterrows(), total=len(df)):
    nome_crop = row["crop"]
    caption = row["caption"]

    try:
        crop_original_path = ORIGINAL_CROPS_DIR / nome_crop
        nome_base = Path(nome_crop).stem
        crop_reconstruido_path = RECONSTRUCTED_CROPS_DIR / f"reconstructed_detalhado_crop_{nome_base}.png"

        if not crop_original_path.exists() or not crop_reconstruido_path.exists():
            print(f"[Aviso] Arquivo não encontrado: {nome_crop}")
            continue

        # Pré-processa imagens
        img1 = preprocess(Image.open(crop_original_path).convert("RGB")).unsqueeze(0).to(device)
        img2 = preprocess(Image.open(crop_reconstruido_path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            feat1 = model.encode_image(img1)
            feat2 = model.encode_image(img2)
            score = torch.nn.functional.cosine_similarity(feat1, feat2).item()

        resultados.append({
            "crop": nome_crop,
            "caption": caption,
            "score_clip": round(score, 4)
        })

    except Exception as e:
        print(f"[Erro] {nome_crop}: {e}")

# 💾 Salvar resultado
df_out = pd.DataFrame(resultados)
df_out.to_csv(OUT_CSV, index=False)

print(f"Similaridade CLIP (crops) salva em: {OUT_CSV}")

