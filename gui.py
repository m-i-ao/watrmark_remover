import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import os
from threading import Thread
from image_processing import process_image

class WatermarkRemoverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Watermark Remover")
        self.root.geometry("1200x700")

        # Переменные
        self.image_paths = []
        self.selected_images = {}
        self.input_folder = ""
        self.output_folder = ""
        self.current_image = None
        self.processed_image = None
        self.canvas_image = None
        self.canvas_processed_image = None
        self.start_point = None
        self.rect = None
        self.selection = None
        self.mode = tk.StringVar(value="auto")

        # GUI элементы
        self.setup_gui()

    def setup_gui(self):
        # Кнопка выбора папки с изображениями
        tk.Button(self.root, text="Выбрать папку с изображениями", command=self.select_input_folder).pack(pady=5)

        # Кнопка выбора папки для сохранения
        tk.Button(self.root, text="Выбрать папку для сохранения", command=self.select_output_folder).pack(pady=5)

        # Режим удаления
        tk.Radiobutton(self.root, text="Автоматическое удаление", variable=self.mode, value="auto").pack()
        tk.Radiobutton(self.root, text="Ручное удаление", variable=self.mode, value="manual").pack()

        # Основной фрейм для предпросмотра и списка
        main_frame = tk.Frame(self.root)
        main_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        # Фрейм для предпросмотра
        preview_frame = tk.Frame(main_frame)
        preview_frame.pack(side=tk.LEFT, padx=10)

        # Канвас для исходного изображения
        self.canvas = tk.Canvas(preview_frame, width=300, height=300, bg="white")
        self.canvas.pack(pady=5)
        self.canvas.bind("<Button-1>", self.start_selection)
        self.canvas.bind("<B1-Motion>", self.update_selection)
        self.canvas.bind("<ButtonRelease-1>", self.end_selection)

        # Канвас для обработанного изображения
        self.canvas_processed = tk.Canvas(preview_frame, width=300, height=300, bg="white")
        self.canvas_processed.pack(pady=5)

        # Фрейм для анализа папки (справа от предпросмотра)
        folder_frame = tk.Frame(main_frame)
        folder_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        # Кнопка анализа папки
        tk.Button(folder_frame, text="Анализировать папку", command=self.analyze_folder).pack(pady=5)

        # Кнопка выбора всех изображений
        tk.Button(folder_frame, text="Выбрать все", command=self.toggle_select_all).pack(pady=5)

        # Канвас для списка изображений со скроллбаром
        self.image_list_canvas = tk.Canvas(folder_frame, width=300, height=400)
        self.image_list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Скроллбар
        scrollbar = ttk.Scrollbar(folder_frame, orient=tk.VERTICAL, command=self.image_list_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_list_canvas.configure(yscrollcommand=scrollbar.set)

        # Фрейм для списка изображений
        self.image_list_frame = tk.Frame(self.image_list_canvas)
        self.image_list_canvas.create_window((0, 0), window=self.image_list_frame, anchor="nw")
        self.image_list_frame.bind("<Configure>", lambda e: self.image_list_canvas.configure(scrollregion=self.image_list_canvas.bbox("all")))

        # Прогресс-бары
        self.total_progress = ttk.Progressbar(self.root, length=300, mode="determinate")
        self.total_progress.pack(pady=5)
        self.image_progress = ttk.Progressbar(self.root, length=300, mode="determinate")
        self.image_progress.pack(pady=5)

        # Кнопка обработки
        tk.Button(self.root, text="Удалить водяные знаки", command=self.process_images).pack(pady=10)

    def select_input_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.input_folder = folder
            # Не анализируем папку автоматически

    def select_output_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_folder = folder

    def analyze_folder(self):
        if not self.input_folder:
            messagebox.showerror("Ошибка", "Выберите папку с изображениями!")
            return

        # Очистка списка
        for widget in self.image_list_frame.winfo_children():
            widget.destroy()
        self.image_paths = []
        self.selected_images = {}

        # Сбор изображений
        for file in os.listdir(self.input_folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.image_paths.append(os.path.join(self.input_folder, file))
                self.selected_images[file] = tk.BooleanVar(value=True)

        # Отображение списка с галочками
        for file in self.image_paths:
            file_name = os.path.basename(file)
            frame = tk.Frame(self.image_list_frame)
            frame.pack(fill=tk.X, pady=2)
            tk.Checkbutton(frame, variable=self.selected_images[file_name]).pack(side=tk.LEFT)
            tk.Label(frame, text=file_name).pack(side=tk.LEFT)

        if self.image_paths:
            self.load_image(0)

    def toggle_select_all(self):
        select_all = not all(var.get() for var in self.selected_images.values())
        for var in self.selected_images.values():
            var.set(select_all)

    def load_image(self, index):
        if index < len(self.image_paths):
            img_path = self.image_paths[index]
            img = Image.open(img_path)
            img = img.resize((300, 300), Image.Resampling.LANCZOS)
            self.current_image = ImageTk.PhotoImage(img)
            if self.canvas_image:
                self.canvas.delete(self.canvas_image)
            self.canvas_image = self.canvas.create_image(150, 150, image=self.current_image)

    def load_processed_image(self, img):
        img = Image.fromarray(img)
        img = img.resize((300, 300), Image.Resampling.LANCZOS)
        self.processed_image = ImageTk.PhotoImage(img)
        if self.canvas_processed_image:
            self.canvas_processed.delete(self.canvas_processed_image)
        self.canvas_processed_image = self.canvas_processed.create_image(150, 150, image=self.processed_image)

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

    def process_images(self):
        if not self.input_folder:
            messagebox.showerror("Ошибка", "Выберите папку с изображениями!")
            return
        if not self.output_folder:
            messagebox.showerror("Ошибка", "Выберите папку для сохранения!")
            return

        selected_paths = [path for path in self.image_paths
                         if self.selected_images[os.path.basename(path)].get()]
        if not selected_paths:
            messagebox.showerror("Ошибка", "Выберите изображения для обработки!")
            return

        def run_processing():
            self.total_progress["maximum"] = len(selected_paths)
            for i, img_path in enumerate(selected_paths):
                self.image_progress["value"] = 0
                self.load_image(i)  # Обновляем предпросмотр исходного изображения

                if self.mode.get() == "auto":
                    processed_img = process_image(img_path)
                else:
                    if self.selection:
                        x1, y1, x2, y2 = self.selection
                        # Масштабирование координат
                        img = Image.open(img_path)
                        x1 = int(x1 * img.width / 300)
                        y1 = int(y1 * img.height / 300)
                        x2 = int(x2 * img.width / 300)
                        y2 = int(y2 * img.height / 300)
                        processed_img = process_image(img_path, manual_coords=(x1, y1, x2, y2))
                    else:
                        continue

                # Обновляем предпросмотр обработанного изображения
                self.load_processed_image(processed_img)

                # Сохранение
                output_path = os.path.join(self.output_folder, os.path.basename(img_path))
                Image.fromarray(processed_img).save(output_path)

                self.image_progress["value"] = 100
                self.total_progress["value"] = i + 1
                self.root.update()

            messagebox.showinfo("Успех", "Обработка завершена!")

        Thread(target=run_processing).start()