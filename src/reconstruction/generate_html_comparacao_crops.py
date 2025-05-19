import pandas as pd
from pathlib import Path

# Diretórios
ORIGINAL_DIR = Path("data")  # imagem original completa
CROP_DIR = Path("results/yolo/crops")
RECONSTRUCTED_DIR = Path("results/reconstructions/from_crops")
CSV = Path("results/reconstructions/clip_similarity_crops.csv")
OUT_HTML = Path("results/reconstructions/visual_comparacao_crops.html")

# Lê o CSV com captions e scores
df = pd.read_csv(CSV)

# Início do HTML
html = """
<!DOCTYPE html>
<html lang="en"> 
<head>
  <meta charset="UTF-8">
  <title>Comparação de Crops – Original vs. Reconstruído</title>
  <style>
    body { font-family: Arial, sans-serif; padding: 20px; }
    table { width: 100%; border-collapse: collapse; }
    td, th { border: 1px solid #ccc; padding: 10px; text-align: center; }
    img { max-width: 150px; max-height: 150px; }
    th { background-color: #f4f4f4; }
  </style>
</head>
<body>
<h2>Comparação de Crops – Original × Reconstruído com Caption e Similaridade</h2>
<table>
  <tr>
    <th>Imagem Original</th>
    <th>Crop Original</th>
    <th>Caption</th>
    <th>Reconstruído</th>
    <th>Score CLIP</th>
  </tr>
"""

for _, row in df.iterrows():
    crop = row["crop"]
    caption = row["caption"]
    score = row["score_clip"]

    # Nome base da imagem original
    nome_base = crop.split("_crop")[0]
    extensao = Path(crop).suffix  # .jpeg, .png...


    # Caminhos das imagens
    #original_img = f"../../data/{nome_base}{extensao.upper()}"
    original_img = f"../../data/{nome_base}.JPG"
    crop_img = f"../../results/yolo/crops/{crop}"
    recon_img = f"../../results/reconstructions/from_crops/reconstructed_detalhado_crop_{Path(crop).stem}.png"

    html += f"""
    <tr>
      <td><img src="{original_img}" alt="{nome_base}"></td>
      <td><img src="{crop_img}" alt="{crop}"></td>
      <td>{caption}</td>
      <td><img src="{recon_img}" alt="recon_{crop}"></td>
      <td><strong>{score:.4f}</strong></td>
    </tr>
    """

html += """
</table>
</body>
</html>
"""

# Salva HTML
OUT_HTML.write_text(html, encoding="utf-8")
print(f"HTML salvo em: {OUT_HTML}")