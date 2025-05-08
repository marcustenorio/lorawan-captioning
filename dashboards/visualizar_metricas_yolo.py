# dashboards/visualizar_metricas_yolo.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import base64
from pathlib import Path
from PIL import Image
from io import BytesIO

# ⚙️ Configuração da página
st.set_page_config(page_title="Métricas YOLOv5", layout="wide")

# 📄 Caminho do CSV
csv_path = Path("results/yolo/metricas_yolo.csv")
crops_dir = Path("results/yolo/crops")

# ❌ Verifica se o arquivo existe
if not csv_path.exists():
    st.error("❌ Arquivo 'metricas_yolo.csv' não encontrado em 'results/yolo/'.")
    st.stop()

# 📊 Carrega dados
df = pd.read_csv(csv_path)

# 🧭 Título
st.title("📊 Dashboard de Métricas – YOLOv5")
st.markdown("Análise interativa com visualização dos recortes detectados.")

# 🔎 Filtro por status
status_opcao = st.multiselect(
    "Filtrar por status:",
    options=df['status'].unique(),
    default=list(df['status'].unique())
)
df_filtrado = df[df['status'].isin(status_opcao)]

# 📋 Tabela de métricas com seleção
st.subheader("🔍 Tabela interativa")
imagem_selecionada = st.selectbox("Selecione uma imagem para ver os crops detectados:", df_filtrado['imagem'])

# 📊 Gráfico: objetos por imagem
st.subheader("📦 Nº de objetos detectados por imagem")
fig1, ax1 = plt.subplots(figsize=(10, 4))
sns.barplot(data=df_filtrado, x="imagem", y="objetos_detectados", ax=ax1, palette="Blues_d")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
st.pyplot(fig1)

# ⏱️ Gráfico: tempo de processamento
st.subheader("⏱️ Tempo de processamento por imagem")
fig2, ax2 = plt.subplots(figsize=(10, 4))
sns.barplot(data=df_filtrado, x="imagem", y="tempo_processamento_s", ax=ax2, palette="Oranges")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
st.pyplot(fig2)

# 🔄 Dispersão: objetos vs tempo
st.subheader("🔄 Correlação: Nº de objetos x Tempo")
fig3, ax3 = plt.subplots(figsize=(6, 4))
sns.scatterplot(data=df_filtrado, x="objetos_detectados", y="tempo_processamento_s", hue="status", s=100, ax=ax3)
st.pyplot(fig3)

# ⚠️ Falhas
num_falhas = df[df['status'].isin(['falha', 'erro', 'erro_leitura'])].shape[0]
if num_falhas > 0:
    st.warning(f"⚠️ {num_falhas} imagem(ns) com falha ou erro.")

# 🖼️ Exibição dos crops
st.subheader(f"🖼️ Recortes detectados: {imagem_selecionada}")

# 🔍 Busca todos os crops da imagem
prefix = Path(imagem_selecionada).stem + "_"
crops = sorted([f for f in crops_dir.glob(f"{prefix}*") if f.suffix in ['.jpg', '.jpeg', '.png']])

if not crops:
    st.info("Nenhum crop detectado ou imagem sem objetos.")
else:
    cols = st.columns(min(len(crops), 5))
    for i, crop_path in enumerate(crops):
        with cols[i % len(cols)]:
            image = Image.open(crop_path)
            st.image(image, caption=crop_path.name, use_column_width=True)
