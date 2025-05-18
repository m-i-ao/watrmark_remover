import cv2
import numpy as np
import pytesseract
from PIL import Image

# Укажите путь к Tesseract, если требуется (например, на Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_watermark(img):
    # Конвертация в градации серого и бинаризация
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Поиск контуров
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Фильтрация контуров по размеру (предполагаем, что водяной знак не слишком маленький)
    watermark_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 50 < w < img.shape[1] * 0.5 and 20 < h < img.shape[0] * 0.5:
            watermark_contours.append((x, y, w, h))

    # Попытка обнаружения текста с помощью Tesseract
    try:
        text_data = pytesseract.image_to_data(Image.fromarray(img), output_type=pytesseract.Output.DICT)
        for i, conf in enumerate(text_data['conf']):
            if int(conf) > 60:  # Высокая уверенность
                x, y, w, h = (text_data['left'][i], text_data['top'][i],
                             text_data['width'][i], text_data['height'][i])
                watermark_contours.append((x, y, w, h))
    except:
        pass  # Если Tesseract не работает, продолжаем с контурами

    # Если контуры найдены, выбираем самый вероятный
    if watermark_contours:
        x, y, w, h = max(watermark_contours, key=lambda c: c[2] * c[3])  # Берем самый большой
        return x, y, x + w, y + h
    else:
        # Если ничего не найдено, возвращаем пустую область
        return 0, 0, 0, 0

def remove_watermark(img, x1, y1, x2, y2):
    if x1 == x2 or y1 == y2:
        return img  # Ничего не найдено, возвращаем оригинал

    # Убедимся, что координаты валидны
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

    # Создаем маску
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    # Улучшенное inpainting с учетом текстуры
    # Сначала применяем размытие для подготовки
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # Используем Telea для начального заполнения
    inpainted = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    
    # Постобработка: восстанавливаем текстуру
    texture = cv2.subtract(img, blurred)
    result = cv2.add(inpainted, texture[y1:y2, x1:x2])

    # Копируем восстановленную область в оригинальное изображение
    img[y1:y2, x1:x2] = result
    return img

def process_image(img_path, manual_coords=None):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if manual_coords:
        x1, y1, x2, y2 = manual_coords
    else:
        x1, y1, x2, y2 = detect_watermark(img)
    
    processed_img = remove_watermark(img, x1, y1, x2, y2)
    return processed_img