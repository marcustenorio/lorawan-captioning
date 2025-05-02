metrics = []

def log_image_metrics(image_name, status, num_detections, elapsed):
    metrics.append({
        'imagem': image_name,
        'status': status,
        'objetos_detectados': num_detections,
        'tempo_processamento_s': round(elapsed, 2)
    })

