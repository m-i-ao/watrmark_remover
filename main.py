import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageTk
import os
from threading import Thread

# Простая CNN для обнаружения водяных знаков
class WatermarkDetectorCNN(nn.Module):
    def __init__(self):
        super(WatermarkDetectorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)  # Предполагается вход 256x256
        self.fc2 = nn.Linear(128, 4)  # Выход: координаты (x1, y1, x2, y2) области водяного знака

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Основной класс приложения
class WatermarkRemoverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Watermark Remover")
        self.root.geometry("800x600")

        # Модель нейронной сети
        self.model = WatermarkDetectorCNN()
        # Для демонстрации: модель не обучена, в реальном проекте нужно обучить
        self.model.eval()

        # Переменные
        self.image_paths = []
        self.current_image = None
        self.canvas_image = None
        self.start_point = None
        self.rect = None
        self.selection = None
        self.mode = tk.StringVar(value="auto")  # auto или manual

        # GUI элементы
        self.setup_gui()

    def setup_gui(self):
        # Кнопка выбора файлов
        tk.Button(self.root, text="Выбрать изображения", command=self.select_files).pack(pady=5)

        # Режим удаления
        tk.Radiobutton(self.root, text="Автоматическое удаление", variable=self.mode, value="auto").pack()
        tk.Radiobutton(self.root, text="Ручное удаление", variable=self.mode, value="manual").pack()

        # Канвас для предпросмотра
        self.canvas = tk.Canvas(self.root, width=400, height=400, bg="white")
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self.start_selection)
        self.canvas.bind("<B1-Motion>", self.update_selection)
        self.canvas.bind("<ButtonRelease-1>", self.end_selection)

        # Прогресс-бары
        self.total_progress = ttk.Progressbar(self.root, length=300, mode="determinate")
        self.total_progress.pack(pady=5)
        self.image_progress = ttk.Progressbar(self.root, length=300, mode="determinate")
        self.image_progress.pack(pady=5)

        # Кнопка обработки
        tk.Button(self.root, text="Удалить водяные знаки", command=self.process_images).pack(pady=10)

    def select_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        self.image_paths = files
        if files:
            self.load_image(0)

    def load_image(self, index):
        if index < len(self.image_paths):
            img_path = self.image_paths[index]
            img = Image.open(img_path)
            img = img.resize((400, 400), Image.Resampling.LANCZOS)
            self.current_image = ImageTk.PhotoImage(img)
            if self.canvas_image:
                self.canvas.delete(self.canvas_image)
            self.canvas_image = self.canvas.create_image(200, 200, image=self.current_image)

    def start_selection(self, event):
        if self.mode.get() == "manual":
            self.start_point = (event.x, event.y)
            if self.rect:
                self.canvas.delete(self.rect)
            self.rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="red")

    def update_selection(self, event):
        if self.mode.get() == "manual" and self.start_point:
            self.canvas.coords(self.rect, self.start_point[0], self.start_point[1], event.x, event.y)

    def end_selection(self, event):
        if self.mode.get() == "manual" and self.start_point:
            self.selection = (self.start_point[0], self.start_point[1], event.x, event.y)
            self.start_point = None

    def detect_watermark(self, img):
        # Подготовка изображения для нейронки
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        with torch.no_grad():
            coords = self.model(img_tensor)
        x1, y1, x2, y2 = coords[0].numpy()
        return int(x1), int(y1), int(x2), int(y2)

    def remove_watermark(self, img, x1, y1, x2, y2):
        # Простое удаление: замена области средним цветом
        mask = np.zeros_like(img)
        mask[y1:y2, x1:x2] = 1
        blurred = cv2.inpaint(img, mask[:, :, 0], inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return blurred

    def process_images(self):
        if not self.image_paths:
            messagebox.showerror("Ошибка", "Выберите изображения!")
            return

        def run_processing():
            self.total_progress["maximum"] = len(self.image_paths)
            for i, img_path in enumerate(self.image_paths):
                self.image_progress["value"] = 0
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img, (256, 256))

                if self.mode.get() == "auto":
                    x1, y1, x2, y2 = self.detect_watermark(img_resized)
                    # Масштабирование координат
                    x1 = int(x1 * img.shape[1] / 256)
                    y1 = int(y1 * img.shape[0] / 256)
                    x2 = int(x2 * img.shape[1] / 256)
                    y2 = int(y2 * img.shape[0] / 256)
                else:
                    if self.selection:
                        x1, y1, x2, y2 = self.selection
                        # Масштабирование координат с канваса на оригинальное изображение
                        x1 = int(x1 * img.shape[1] / 400)
                        y1 = int(y1 * img.shape[0] / 400)
                        x2 = int(x2 * img.shape[1] / 400)
                        y2 = int(y2 * img.shape[0] / 400)
                    else:
                        continue

                processed_img = self.remove_watermark(img, x1, y1, x2, y2)
                output_path = os.path.splitext(img_path)[0] + "_no_watermark.jpg"
                cv2.imwrite(output_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))

                self.image_progress["value"] = 100
                self.total_progress["value"] = i + 1
                self.root.update()

            messagebox.showinfo("Успех", "Обработка завершена!")

        Thread(target=run_processing).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = WatermarkRemoverApp(root)
    root.mainloop()