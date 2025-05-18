#!/bin/bash

echo "🧹 Limpando arquivos gerados pelo pipeline (csv, json, png)..."

# Diretórios-alvo
TARGET_DIRS=(
  "results/blip/captions"
  "results/blip/full_image"
  "results/reconstructions/from_caption"
  "results/reconstructions/from_crops"
  "results/reconstructions"
)

# Extensões a remover
EXTENSIONS=("csv" "json" "png")

for dir in "${TARGET_DIRS[@]}"; do
  if [ -d "$dir" ]; then
    echo "→ Limpando $dir"
    for ext in "${EXTENSIONS[@]}"; do
      find "$dir" -type f -name "*.${ext}" -exec rm -v {} \;
    done
  else
    echo "⚠️ Diretório não encontrado: $dir"
  fi
done

echo "✅ Limpeza finalizada."

