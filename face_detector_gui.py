import cv2
import dlib
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import os

class FaceDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Детектор лиц - Face Detector")
        self.root.geometry("800x600")
        self.root.configure(bg='#2c3e50')
        
        # Переменные для управления камерой
        self.cap = None
        self.is_running = False
        self.detector = None
        self.predictor = None
        
        # Настройки детекции
        self.show_landmarks = tk.BooleanVar(value=True)
        self.show_rectangles = tk.BooleanVar(value=True)
        
        self.setup_ui()
        self.load_models()
        
    def setup_ui(self):
        # Заголовок
        title_label = tk.Label(
            self.root, 
            text="Детектор лиц с использованием OpenCV и Dlib",
            font=("Arial", 16, "bold"),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=10)
        
        # Фрейм для видео
        self.video_frame = tk.Frame(self.root, bg='#34495e', relief=tk.RAISED, bd=2)
        self.video_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        # Лейбл для отображения видео
        self.video_label = tk.Label(
            self.video_frame, 
            text="Нажмите 'Запустить камеру' для начала работы",
            font=("Arial", 12),
            bg='#34495e',
            fg='white'
        )
        self.video_label.pack(expand=True)
        
        # Фрейм для настроек
        settings_frame = tk.Frame(self.root, bg='#2c3e50')
        settings_frame.pack(pady=10)
        
        # Чекбоксы для настроек
        landmarks_check = tk.Checkbutton(
            settings_frame,
            text="Показывать ключевые точки",
            variable=self.show_landmarks,
            bg='#2c3e50',
            fg='white',
            selectcolor='#34495e',
            font=("Arial", 10)
        )
        landmarks_check.pack(side=tk.LEFT, padx=10)
        
        rectangles_check = tk.Checkbutton(
            settings_frame,
            text="Показывать прямоугольники",
            variable=self.show_rectangles,
            bg='#2c3e50',
            fg='white',
            selectcolor='#34495e',
            font=("Arial", 10)
        )
        rectangles_check.pack(side=tk.LEFT, padx=10)
        
        # Фрейм для кнопок
        button_frame = tk.Frame(self.root, bg='#2c3e50')
        button_frame.pack(pady=20)
        
        # Кнопка запуска камеры
        self.start_button = tk.Button(
            button_frame,
            text="Запустить камеру",
            command=self.start_camera,
            bg='#27ae60',
            fg='white',
            font=("Arial", 12, "bold"),
            padx=20,
            pady=10,
            relief=tk.RAISED,
            bd=3
        )
        self.start_button.pack(side=tk.LEFT, padx=10)
        
        # Кнопка остановки камеры
        self.stop_button = tk.Button(
            button_frame,
            text="Остановить камеру",
            command=self.stop_camera,
            bg='#e74c3c',
            fg='white',
            font=("Arial", 12, "bold"),
            padx=20,
            pady=10,
            relief=tk.RAISED,
            bd=3,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=10)
        
        # Информационная панель
        info_frame = tk.Frame(self.root, bg='#2c3e50')
        info_frame.pack(pady=10)
        
        self.info_label = tk.Label(
            info_frame,
            text="Статус: Готов к работе",
            font=("Arial", 10),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        self.info_label.pack()
        
    def load_models(self):
        """Загрузка моделей Dlib"""
        try:
            # Проверяем наличие файла модели
            model_path = "shape_predictor_68_face_landmarks.dat"
            if not os.path.exists(model_path):
                messagebox.showerror(
                    "Ошибка", 
                    f"Файл модели {model_path} не найден!\n"
                    "Убедитесь, что файл находится в той же папке, что и программа."
                )
                return False
                
            # Загружаем детектор лиц
            self.detector = dlib.get_frontal_face_detector()
            
            # Загружаем предиктор ключевых точек
            self.predictor = dlib.shape_predictor(model_path)
            
            self.info_label.config(text="Статус: Модели загружены успешно")
            return True
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка загрузки моделей: {str(e)}")
            return False
    
    def start_camera(self):
        """Запуск камеры"""
        if not self.detector or not self.predictor:
            if not self.load_models():
                return
                
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Ошибка", "Не удалось открыть камеру!")
                return
                
            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.info_label.config(text="Статус: Камера работает")
            
            # Запускаем обработку видео в отдельном потоке
            self.video_thread = threading.Thread(target=self.process_video)
            self.video_thread.daemon = True
            self.video_thread.start()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка запуска камеры: {str(e)}")
    
    def stop_camera(self):
        """Остановка камеры"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
            
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.info_label.config(text="Статус: Камера остановлена")
        
        # Очищаем видео лейбл
        self.video_label.config(image='', text="Нажмите 'Запустить камеру' для начала работы")
    
    def process_video(self):
        """Обработка видеопотока"""
        while self.is_running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Обрабатываем кадр
            processed_frame = self.detect_faces(frame)
            
            # Конвертируем для отображения в tkinter
            self.display_frame(processed_frame)
    
    def detect_faces(self, frame):
        """Детекция лиц на кадре"""
        # Преобразуем в оттенки серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Обнаруживаем лица
        faces = self.detector(gray)
        
        faces_count = len(faces)
        
        for face in faces:
            # Рисуем прямоугольник вокруг лица
            if self.show_rectangles.get():
                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Определяем ключевые точки
            if self.show_landmarks.get():
                landmarks = self.predictor(gray, face)
                
                # Рисуем ключевые точки
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        
        # Обновляем информацию о количестве лиц
        self.root.after(0, lambda: self.info_label.config(
            text=f"Статус: Камера работает | Обнаружено лиц: {faces_count}"
        ))
        
        return frame
    
    def display_frame(self, frame):
        """Отображение кадра в GUI"""
        # Изменяем размер кадра для отображения
        height, width = frame.shape[:2]
        max_width = 640
        max_height = 480
        
        if width > max_width or height > max_height:
            scale = min(max_width/width, max_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Конвертируем BGR в RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Конвертируем в PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Конвертируем в PhotoImage для tkinter
        photo = ImageTk.PhotoImage(pil_image)
        
        # Обновляем лейбл в главном потоке
        self.root.after(0, lambda: self.update_video_label(photo))
    
    def update_video_label(self, photo):
        """Обновление видео лейбла"""
        self.video_label.config(image=photo, text='')
        self.video_label.image = photo  # Сохраняем ссылку
    
    def on_closing(self):
        """Обработка закрытия окна"""
        self.stop_camera()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = FaceDetectorGUI(root)
    
    # Обработка закрытия окна
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    root.mainloop()

if __name__ == "__main__":
    main()

