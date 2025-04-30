# Compactação Semântica e Reconstrução de Imagens via LoRaWAN

Este projeto explora o uso de visão computacional e IA para simular a transmissão de imagens em redes de baixa largura de banda (LoRaWAN), utilizando captioning e reconstrução com modelos generativos.

## 🔧 Pipeline Principal
1. YOLOv5 para detecção de objetos
2. BLIP para geração de descrições semânticas (captioning)
3. Simulação de envio de texto via LoRaWAN (NS-3)
4. Reconstrução da imagem via Stable Diffusion
5. Avaliação de fidelidade com CLIP

## 📁 Estrutura do Projeto
- `src/`: scripts organizados por módulo
- `notebooks/`: experimentos exploratórios
- `results/`: outputs e reconstruções
- `data/`: imagens utilizadas nos testes

## 🚀 Requisitos
Instale as dependências:
```bash
pip install -r requirements.txt

