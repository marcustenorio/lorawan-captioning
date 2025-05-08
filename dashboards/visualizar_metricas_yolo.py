import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Métricas YOLOv5", layout="wide")

# Carrega CSV
csv_path = Path("results/yolo/metricas_yolo.csv")
if not csv_path.exists():
    st.error("Arquivo de métricas não encontrado.")
    st.stop()

df = pd.read_csv(csv_path)

st.title("📊 Dashboard de Métricas – YOLOv5")
st.markdown("Visualização interativa das métricas de detecção")

# Filtro por status
status_opcao = st.multiselect("Filtrar por status", options=df['status'].unique(), default=list(df['status'].unique()))
df_filtrado = df[df['status'].isin(status_opcao)]

# Tabela interativa
st.subheader("🔍 Tabela de métricas")
st.dataframe(df_filtrado.sort_values(by="objetos_detectados", ascending=False), use_container_width=True)

# Gráfico: Objetos por imagem
st.subheader("📦 Nº de objetos detectados por imagem")
fig1, ax1 = plt.subplots(figsize=(10, 4))
sns.barplot(data=df_filtrado, x="imagem", y="objetos_detectados", ax=ax1, palette="Blues_d")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
st.pyplot(fig1)

# Gráfico: Tempo de processamento
st.subheader("⏱️ Tempo de processamento por imagem")
fig2, ax2 = plt.subplots(figsize=(10, 4))
sns.barplot(data=df_filtrado, x="imagem", y="tempo_processamento_s", ax=ax2, palette="Oranges")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
st.pyplot(fig2)

# Gráfico: Dispersão
st.subheader("🔄 Correlação: Nº de objetos x Tempo de processamento")
fig3, ax3 = plt.subplots(figsize=(6, 4))
sns.scatterplot(data=df_filtrado, x="objetos_detectados", y="tempo_processamento_s", hue="status", s=100, ax=ax3)
st.pyplot(fig3)

# Contador de falhas
num_falhas = df[df['status'].isin(['falha', 'erro', 'erro_leitura'])].shape[0]
st.warning(f"⚠️ {num_falhas} imagem(ns) com falha ou erro.")
