import torch
from torchvision import models, transforms
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from pathlib import Path
import cv2

# 🔧 Carrega um modelo ResNet50 pré-treinado
resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()

# 🎯 Inicializa o GradCAM na última camada convolucional ('layer4')
cam_extractor = GradCAM(model=resnet_model, target_layer="layer4", input_shape=(1, 3, 224, 224))

# 🧰 Preprocessamento da imagem para ResNet50
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def gerar_mapa_explicativo_resnet(img_bgr, nome_arquivo, pasta_destino):
    try:
        # Converte a imagem de BGR para RGB e para PIL
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Transforma para tensor e garante que o gradiente seja calculado
        input_tensor = transform(pil_img).unsqueeze(0)
        input_tensor.requires_grad_(True)  # habilita gradiente

        # Faz o forward com cálculo de gradiente
        output = resnet_model(input_tensor)
        
        # Seleciona a classe prevista (top-1)
        class_idx = output.squeeze().argmax().item()

        # Gera o mapa de ativação (CAM) passando os scores (logits)
        # Aqui não usamos torch.no_grad() para que os hooks capturem os gradientes
        activation_map = cam_extractor(class_idx, scores=output)[0]

        # Sobrepõe o mapa de ativação à imagem original
        resultado_img = overlay_mask(pil_img, to_pil_image(activation_map, mode='F'), alpha=0.6)

        # Salva o resultado
        Path(pasta_destino).mkdir(parents=True, exist_ok=True)
        resultado_img.save(Path(pasta_destino) / f"expl_{nome_arquivo}")

    except Exception as e:
        print(f"[Erro] Explicabilidade: {e}")
