import cv2
import numpy as np
import torch

def detect_watermark(img, model):
    # Подготовка изображения для нейронки
    img_resized = cv2.resize(img, (256, 256))
    img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    with torch.no_grad():
        coords = model(img_tensor)
    x1, y1, x2, y2 = coords[0].numpy()
    
    # Масштабирование координат
    x1 = int(x1 * img.shape[1] / 256)
    y1 = int(y1 * img.shape[0] / 256)
    x2 = int(x2 * img.shape[1] / 256)
    y2 = int(y2 * img.shape[0] / 256)

    # Уточнение области с помощью анализа контуров
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Находим контур, ближайший к предсказанной области
    best_contour = None
    min_dist = float('inf')
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x + w // 2, y + h // 2
        center_pred = ((x1 + x2) // 2, (y1 + y2) // 2)
        dist = ((cx - center_pred[0]) ** 2 + (cy - center_pred[1]) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            best_contour = contour

    if best_contour is not None:
        x, y, w, h = cv2.boundingRect(best_contour)
        x1, y1, x2, y2 = x, y, x + w, y + h

    return x1, y1, x2, y2

def remove_watermark(img, x1, y1, x2, y2):
    # Убедимся, что координаты валидны
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    
    # Создаем маску для области водяного знака
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    
    # Используем inpainting для удаления водяного знака
    result = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
    return result

def process_image(img_path, model, manual_coords=None):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if manual_coords:
        x1, y1, x2, y2 = manual_coords
    else:
        x1, y1, x2, y2 = detect_watermark(img, model)
    
    processed_img = remove_watermark(img, x1, y1, x2, y2)
    return processed_img