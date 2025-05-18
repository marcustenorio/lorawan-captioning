# src/simulation/export_for_ns3.py

import pandas as pd
from pathlib import Path

CSV_PATH = Path("results/blip/captions/captions_crops.csv")
OUT_DIR = Path("results/ns3/input")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

for i, row in df.iterrows():
    caption = row["caption"]
    with open(OUT_DIR / f"node_{i}.txt", "w") as f:
        f.write(caption)

